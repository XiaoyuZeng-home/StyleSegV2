#####REG+SEG#####
import argparse
import os
import json
import re
import datetime
import time
from timeit import default_timer

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--base_network', type=str, default='RWUNET_v1',
                    help='Specifies the base network (either VTN or VoxelMorph)')
parser.add_argument('-n', '--n_cascades', type=int, default=1,
                    help='Number of cascades')
parser.add_argument('-r', '--rep', type=int, default=1,
                    help='Number of times of shared-weight cascading')
parser.add_argument('-g', '--gpu', type=str, default='1',
                    help='Specifies gpu device(s)')
parser.add_argument('-c', '--checkpoint', type=str, default=None,##Apr04-0957
                    help='Specifies a previous checkpoint to start with')
parser.add_argument('-d', '--dataset', type=str, default="datasets/OASIS/brain.json",
                    help='Specifies a data config')
parser.add_argument('--batch', type=int, default=1,
                    help='Number of image pairs per batch')


####numbers of iteration of reg-seg 
parser.add_argument('--iter_num', type=int, default=3, help='Number of iterations')
parser.add_argument('--model_dir', default='./weights')
parser.add_argument('--reg_round', type=int, default=2000, help='Number of batches per reg_epoch')
parser.add_argument('--seg_round', type=int, default=1000,help='Number of batches per seg_epoch')
parser.add_argument('--reg_lr', type=float, default=1e-4)
parser.add_argument('--seg_lr', type=float, default=1e-3)
parser.add_argument('--scheme', type=str, default=None, help='chose reg、seg、reg_supervise')
parser.add_argument('--pretrained_type', type=str, default=None, help='chose seg、reg_supervise')

parser.add_argument('--epochs', type=float, default=10,
                    help='Number of epochs')
parser.add_argument('--fast_reconstruction', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--val_steps', type=int, default=200)#200
parser.add_argument('--net_args', type=str, default='')
parser.add_argument('--data_args', type=str, default='')
parser.add_argument('--clear_steps', action='store_true')
parser.add_argument('--finetune', type=str, default=None)
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--logs', type=str, default='')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import tensorflow as tf
import tflearn
import keras

import network
import data_util.liver
import data_util.brain
from data_util.data import Split

#Loading model
def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    print( 'tf.global_variables(): ',tf.global_variables())
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                    if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)
    
def update_model_dict(models_list,scheme,index,lr,iterationSize):
    key = list(models_list.keys())[-1]
    model_pth = train(models_list[key],scheme,lr,iterationSize)
    iter_key = scheme[:3] + '_model_iter_' + str(index)
    models_list[iter_key] = model_pth
    print(models_list)
    return models_list

#trainig from stracth
def iter_train(models_list):
    for i in range(args.iter_num):
        if i == 0: 
            models_list = update_model_dict(models_list,'reg',i,args.reg_lr,args.reg_round)
        else:
            models_list = update_model_dict(models_list,'reg_supervise',i,args.reg_lr,args.reg_round)
        tf.reset_default_graph()
        models_list = update_model_dict(models_list,'seg', i, args.seg_lr,args.seg_round)
        tf.reset_default_graph()
    import json
    path = os.path.join(args.model_dir,'model_list.json')
    with open(path, "w") as f:
        f.write(json.dumps(models_list, ensure_ascii=False, indent=4, separators=(',', ':')))
        
def iter_train_embedding(models_list,pretrained_type):
    '''
    type:pretrained type
    '''
    for i in range(args.iter_num):
        if pretrained_type=='reg_supervise':
            models_list = update_model_dict(models_list,'seg',i,args.seg_lr,args.seg_round)
            tf.reset_default_graph()
            models_list = update_model_dict(models_list,'reg_supervise', i, args.reg_lr,args.reg_round)
            tf.reset_default_graph()
        if pretrained_type=='seg':
            models_list = update_model_dict(models_list,'reg_supervise',i,args.reg_lr,args.reg_round)
            tf.reset_default_graph()
            models_list = update_model_dict(models_list,'seg', i, args.seg_lr,args.seg_round)
            tf.reset_default_graph()
        if pretrained_type==None:
            iter_train(models_list)
    import json
    path = os.path.join(args.model_dir,'model_list.json')
    with open(path, "w") as f:
        f.write(json.dumps(models_list, ensure_ascii=False, indent=4, separators=(',', ':')))
    
def train(checkpoint,scheme,lr,iterationSize):
    repoRoot = os.path.dirname(os.path.realpath(__file__))
    print('repoRoot:',repoRoot)

    if args.finetune is not None:
        args.clear_steps = True

    batchSize = args.batch

    gpus = 0 if args.gpu == '-1' else len(args.gpu.split(','))

    Framework = network.FrameworkUnsupervised
    Framework.net_args['base_network'] = args.base_network
    Framework.net_args['n_cascades'] = args.n_cascades
    Framework.net_args['rep'] = args.rep
    if scheme == "seg":
        Framework.net_args['augmentation'] = "identity"
    else:
        Framework.net_args['augmentation'] = None
    Framework.net_args['scheme'] = scheme
    Framework.net_args.update(eval('dict({})'.format(args.net_args)))
    with open(os.path.join(args.dataset), 'r') as f:
        cfg = json.load(f)
        image_size = cfg.get('image_size', [160, 160, 160])
        image_type = cfg.get('image_type')
    framework = Framework(devices=gpus, image_size=image_size, segmentation_class_value=cfg.get('segmentation_class_value', None), fast_reconstruction = args.fast_reconstruction)
    Dataset = eval('data_util.{}.Dataset'.format(image_type))
    print('Graph built.')

    # load training set and validation set

    def set_tf_keys(feed_dict, **kwargs):
        ret = dict([(k + ':0', v) for k, v in feed_dict.items()])
        ret.update([(k + ':0', v) for k, v in kwargs.items()])
        return ret
    
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
        
    config = tf.ConfigProto(allow_soft_placement = True) 
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES), keep_checkpoint_every_n_hours=5)
        if checkpoint is None:
            steps = 0
            tf.global_variables_initializer().run()
        else:
            if '\\' not in checkpoint and '/' not in checkpoint:
                checkpoint = os.path.join(
                    repoRoot, args.model_dir, checkpoint)
            if os.path.isdir(checkpoint):
                print('checkpoint: ', checkpoint)
                checkpoint = tf.train.latest_checkpoint(checkpoint)

            tf.compat.v1.global_variables_initializer().run()
            
            checkpoints = checkpoint.split(';')

            if args.clear_steps:
                steps = 0
            else:
                steps = int(re.search('model-(\d+)', checkpoints[0]).group(1))


            for cp in checkpoints:
                optimistic_restore(sess, cp)
                for var in tf.global_variables():
                    #if 'deform' in var.name:
                    print('var: ',var)
                if scheme != 'reg_unsupervise':
                    var_feature = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gaffdfrm/feature')
                    var_list = dict(zip(map(lambda x:x.name.replace('feature','deform_stem_0').split(':')[0], var_feature), var_feature))
                    saver_feature = tf.train.Saver(var_list)
                    saver_feature.restore(sess, cp)
                
        data_args = eval('dict({})'.format(args.data_args))
        data_args.update(framework.data_args)
        print('data_args', data_args)
        dataset = Dataset(args.dataset, **data_args)
        if args.finetune is not None:
            if 'finetune-train-%s' % args.finetune in dataset.schemes:
                dataset.schemes[Split.TRAIN] = dataset.schemes['finetune-train-%s' %
                                                               args.finetune]
            if 'finetune-val-%s' % args.finetune in dataset.schemes:
                dataset.schemes[Split.VALID] = dataset.schemes['finetune-val-%s' %
                                                               args.finetune]
            print('train', dataset.schemes[Split.TRAIN])
            print('val', dataset.schemes[Split.VALID])
            
        if scheme == 'seg':
            if_seg=True
        else:
            if_seg=False
        generator = dataset.generator(Split.TRAIN, batch_size=batchSize, loop=True, pair_train=False,  if_seg=if_seg)

        if not args.debug:
            if args.finetune is not None:
                run_id = os.path.basename(os.path.dirname(checkpoint))
                if not run_id.endswith('_ft' + args.finetune):
                    run_id = run_id + '_ft' + args.finetune
            else:
                pad = ''
                retry = 1
                while True:
                    dt = datetime.datetime.now(
                        tz=datetime.timezone(datetime.timedelta(hours=8)))
                    run_id = dt.strftime('%b%d-%H%M') + pad
                    modelPrefix = os.path.join(repoRoot, args.model_dir, run_id)
                    try:
                        os.makedirs(modelPrefix)
                        break
                    except Exception as e:
                        print('Conflict with {}! Retry...'.format(run_id))
                        pad = '_{}'.format(retry)
                        retry += 1
            modelPrefix = os.path.join(repoRoot, args.model_dir, run_id)
            if not os.path.exists(modelPrefix):
                os.makedirs(modelPrefix)
            
            if args.name is not None:
                run_id += '_' + args.name
            if args.logs is None:
                log_dir = 'logs'
            else:
                log_dir = os.path.join('logs', args.logs)
            summary_path = os.path.join(repoRoot, log_dir, run_id)
            if not os.path.exists(summary_path):
                os.makedirs(summary_path)
            summaryWriter = tf.summary.FileWriter(summary_path, sess.graph)
            with open(os.path.join(modelPrefix, 'args.json'), 'w') as fo:
                json.dump(vars(args), fo)

        if args.finetune is not None:
            learningRates = [1e-5 / 2, 1e-5 / 2, 1e-5 / 2, 1e-5 / 4, 1e-5 / 8]
        else:
            learningRates = [1e-4, 1e-4, 1e-4,1e-4, 1e-4 / 2, 1e-4 / 2, 1e-4 / 2, 1e-4 / 4, 1e-4 / 4,1e-4/8]#10 epoc 
            #[1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4 , 1e-4 , 1e-4 , 1e-4 ,1e-4]

        # Training
        def get_lr(steps):
            m = lr / learningRates[0]
            return m * learningRates[steps // iterationSize]

        last_save_stamp = time.time()
        best_dice_score = 0.0
        while True:
            if hasattr(framework, 'get_lr'):
                lr = framework.get_lr(steps, batchSize)
            else:
                lr = get_lr(steps)
            t0 = default_timer()
            fd = next(generator)
            print('fd :',fd['voxelT1'].shape)
            fd.pop('mask', [])
            id1 = fd.pop('id1', [])
            id2 = fd.pop('id2', [])
            t1 = default_timer()
            tflearn.is_training(True, session=sess)
            #写入loss,执行优化
            summ, _ = sess.run([framework.summaryExtra, framework.adamOpt],
                               set_tf_keys(fd, learningRate=lr))

            for v in tf.Summary().FromString(summ).value:
                if v.tag == 'reg_loss':
                    loss = v.simple_value

            steps += 1
            if args.debug or steps % 10 == 0:
                if steps >= args.epochs * iterationSize:
                    break

                if not args.debug:
                    summaryWriter.add_summary(summ, steps)

                if steps % 100 == 0:
                    if hasattr(framework, 'summaryImages'):
                        summ, = sess.run([framework.summaryImages],
                                         set_tf_keys(fd))
                        summaryWriter.add_summary(summ, steps)

                if steps % 50 == 0:
                    print('*%s* ' % run_id,
                          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                          'Steps %d, Total time %.2f, data %.2f%%. Loss %.3e lr %.3e' % (steps,
                                                                                         default_timer() - t0,
                                                                                         (t1 - t0) / (
                                                                                             default_timer() - t0),
                                                                                         loss,
                                                                                         lr),
                          end='\n')

                if args.debug or steps % args.val_steps == 0:
                    val_gen = dataset.generator(
                        Split.VALID, loop=False, batch_size=batchSize,  if_seg= if_seg, pair_train=False)
                    if scheme == 'reg' or scheme== 'reg_supervise' or scheme== 'reg_unsupervise':
                        keys = ['reg_dices_score', 'reg_dices', 'landmark_dist', 'pt_mask', 'jacc_score','ncc_score']
                    else:
                        keys = ['reg_dices_score', 'seg_dices', 'dices1', 'seg_dices_score', 'dices_pseudo']
                        
                    metrics = framework.validate(
                        sess, val_gen,keys=keys, summary=True)

                    val_summ = tf.Summary(value=[
                        tf.Summary.Value(tag='val_' + k, simple_value=v) for k, v in metrics.items()
                    ])
                    if scheme == 'reg' or scheme == 'reg_supervise' or scheme== 'reg_unsupervise':
                        dice_score = metrics['reg_dices_score']
                    else:
                        dice_score = metrics['seg_dices_score']
                    print('dice:',dice_score)#if use segnet,change dice_score1 to dice_score2
                    if dice_score>best_dice_score:
                        best_dice_score = dice_score
                        print('saving best dice sore:{}'.format(best_dice_score))
                        saver.save(sess, os.path.join(modelPrefix, 'model'),global_step=steps,write_meta_graph=False)
                        with open(os.path.join(modelPrefix,'log.txt'),'a+') as f:
                            f.write('saving best '+keys[1]+'_score :{},steps={} \n'.format(best_dice_score,steps))
                    summaryWriter.add_summary(val_summ, steps)
           
    print('Finished.')
    return os.path.join(args.model_dir,run_id)
        
if __name__ == '__main__':
    models_list = {'pretrained_model':args.checkpoint}
    key = list(models_list.keys())[-1]
    print('pretrained model is :',models_list[key])
    #training iteratively
    if args.scheme == None:
        iter_train_embedding(models_list,args.pretrained_type)
    #training sepertately
    if args.scheme == 'seg':
        train(models_list[key],args.scheme,args.seg_lr,args.seg_round)
    else:
        train(models_list[key],args.scheme,args.reg_lr,args.reg_round)
    # if args.scheme == 'reg' or args.scheme == 'reg_supervise':
    #     print('Registration training')
    #     train(models_list[key],args.scheme,args.reg_lr,args.reg_round)
        
    


        
        
