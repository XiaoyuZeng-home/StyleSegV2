import tensorflow as tf
import numpy as np
import tflearn
from tqdm import tqdm
from . import transform
from .utils import MultiGPUs
from .spatial_transformer import Dense3DSpatialTransformer, Fast3DTransformer
from .recursive_cascaded_networks import RecursiveCascadedNetworks
from .recursive_cascaded_networks import mask_class
import h5py
from scipy.ndimage.interpolation import map_coordinates, zoom
import scipy
from .data_augmentation import random_affine

def set_tf_keys(feed_dict, **kwargs):
    ret = dict([(k + ':0', v) for k, v in feed_dict.items()])
    ret.update([(k + ':0', v) for k, v in kwargs.items()])
    return ret


def masked_mean(arr, mask):
    return tf.reduce_sum(arr * mask) / (tf.reduce_sum(mask) + 1e-9)


class FrameworkUnsupervised:
    net_args = {'class': RecursiveCascadedNetworks}
    framework_name = 'gaffdfrm'

    def __init__(self, devices, image_size, segmentation_class_value, validation=False, fast_reconstruction=False):
        network_class = self.net_args.get('class', RecursiveCascadedNetworks)
        self.summaryType = self.net_args.pop('summary', 'basic')
        self.image_size = image_size

        self.reconstruction = Fast3DTransformer() if fast_reconstruction else Dense3DSpatialTransformer()

        # input place holder
        imgT1_fixed = tf.placeholder(dtype=tf.float32, shape=[
                              None]+image_size+[1], name='voxelT1')
        imgT1_float = tf.placeholder(dtype=tf.float32, shape=[
                              None]+image_size+[1], name='atlasT1')

        seg1 = tf.placeholder(dtype=tf.float32, shape=[
                              None]+image_size+[1], name='seg1')
        seg2 = tf.placeholder(dtype=tf.float32, shape=[
                              None]+image_size+[1], name='seg2')
        point1 = tf.placeholder(dtype=tf.float32, shape=[
                                None, None, 3], name='point1')
        point2 = tf.placeholder(dtype=tf.float32, shape=[
                                None, None, 3], name='point2')#task2

        pseudo_label = tf.placeholder(dtype=tf.float32, shape=[
                              None]+image_size+[1], name='pseudo_label')

        bs = tf.shape(imgT1_fixed)[0]
        Img1, augImg2= imgT1_fixed/255 , imgT1_float/255

        aug = self.net_args.pop('augmentation', None)
        if aug is None:
            imgs = imgT1_fixed.shape.as_list()[1:4]
            def composed_flow(shape):
                control_fields = transform.sample_power(
                    -0.4, 0.4, 3, tf.stack([bs, 5, 5, 5, 3])) * (np.array(shape) // 4)
                elastic_flow = transform.free_form_fields(shape, control_fields)
                affine_flow = random_affine(shape)
                return self.reconstruction([affine_flow, elastic_flow]) + elastic_flow

            def augmentation(x,flow):
                if not tflearn.get_training_mode():
                    print('evaluate!!')
                return tf.cond(tflearn.get_training_mode(), lambda: self.reconstruction([x, flow]),lambda: x)
    
            def augmenetation_pts(incoming,flow):
                def aug(incoming,flow):
                    aug_pt = tf.cast(transform.warp_points(
                        flow, incoming), tf.float32)
                    pt_mask = tf.cast(tf.reduce_all(
                        incoming >= 0, axis=-1, keep_dims=True), tf.float32)
                    return aug_pt * pt_mask - (1 - pt_mask)
                return tf.cond(tflearn.get_training_mode(), lambda: aug(incoming,flow), lambda: incoming)
            
            #fixed img augmentation
            augFlow_1 = composed_flow(imgs)
            Img1 = augmentation(Img1,augFlow_1)
            seg1 = augmentation(seg1,augFlow_1)
            point1 = augmenetation_pts(point1,augFlow_1)
            pseudo_label = augmentation(pseudo_label,augFlow_1)

            augImg2 = augImg2
            augSeg2 = seg2
            augPt2 = point2
        elif aug == 'identity':
            augFlow = tf.zeros(
                tf.stack([tf.shape(imgT1_fixed)[0], image_size[0], image_size[1], image_size[2], 3]), dtype=tf.float32)
            augSeg2 = seg2
            augPt2 = point2
        else:
            raise NotImplementedError('Augmentation {}'.format(aug))

        learningRate = tf.placeholder(tf.float32, [], 'learningRate')
        if not validation:
            adamOptimizer = tf.train.AdamOptimizer(learningRate)#AdaBeliefOptimizer(learning_rate=learningRate, epsilon=1e-14, rectify=False)
        self.segmentation_class_value = segmentation_class_value
        scheme = self.net_args.pop('scheme', None)
        self.network = network_class(
            self.framework_name, framework=self, fast_reconstruction=fast_reconstruction, scheme=scheme, **self.net_args)
        net_pls = [Img1,augImg2,seg1, augSeg2, point1, augPt2, pseudo_label]
        if devices == 0:
            with tf.device("/cpu:0"):
                self.predictions = self.network(*net_pls)
                if not validation:
                    var_segment =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="gaffdfrm/seg_stem")
                    self.adamOpt = adamOptimizer.minimize(
                        self.predictions["reg_loss"])
        else:
            gpus = MultiGPUs(devices)
            if validation:
                self.predictions = gpus(self.network, net_pls, scheme=scheme)
            else:
                self.predictions, self.adamOpt = gpus(
                    self.network, net_pls, opt=adamOptimizer, scheme=scheme)
        self.build_summary(self.predictions)

    @property
    def data_args(self):
        return self.network.data_args

    def build_summary(self, predictions):
        self.loss = tf.reduce_mean(predictions['reg_loss'])
        for k in predictions:
            if k.find('loss') != -1:
                tf.summary.scalar(k, tf.reduce_mean(predictions[k]))
        self.summaryOp = tf.summary.merge_all()

        if self.summaryType == 'full':
            tf.summary.scalar('reg_dices_score', tf.reduce_mean(
                self.predictions['reg_dices_score']))
            tf.summary.scalar('landmark_dist', masked_mean(
                self.predictions['landmark_dist'], self.predictions['pt_mask']))
            preds = tf.reduce_sum(
                tf.cast(self.predictions['jacc_score'] > 0, tf.float32))
            tf.summary.scalar('jacc_score', tf.reduce_sum(
                self.predictions['jacc_score']) / (preds + 1e-8))
            self.summaryExtra = tf.summary.merge_all()
        else:
            self.summaryExtra = self.summaryOp
        self.summaryImages1 = tf.summary.image('fixed_img', tf.reshape(self.predictions['image_fixed_T1'][:,96,:,:,0], (1,self.image_size[1],self.image_size[2],1)))
        self.summaryImages2 = tf.summary.image('warped_moving_img', tf.reshape(self.predictions['warped_moving_T1'][:,96,:,:,0], (1,self.image_size[1],self.image_size[2],1)))
        self.summaryImages3 = tf.summary.image('image_float_T1', tf.reshape(self.predictions['image_float_T1'][:,96,:,:,0], (1,self.image_size[1],self.image_size[2],1)))
        self.summaryImages = tf.summary.merge([self.summaryImages1,self.summaryImages2,self.summaryImages3])

    def get_predictions(self, *keys):
        return dict([(k, self.predictions[k]) for k in keys])

    def validate_clean(self, sess, generator, keys=None):
        for fd in generator:
            _ = fd.pop('id1')
            _ = fd.pop('id2')
            _ = sess.run(self.get_predictions(*keys),
                         feed_dict=set_tf_keys(fd))
            
    def fusion_dices(self,candidates, target_seg):
        def compute_dice_coefficient(mask_gt, mask_pred):
            volume_sum = mask_gt.sum() + mask_pred.sum()
            if volume_sum == 0:
                return 0
            volume_intersect = (mask_gt & mask_pred).sum()
            return 2*volume_intersect / volume_sum
        fusion_label_onehot = np.mean(candidates,axis=0,keepdims=False)
        fusion_label = np.argmax(fusion_label_onehot,axis=-1)
        dices = []
        for i in range(fusion_label_onehot.shape[-1]):
            dices.append(compute_dice_coefficient(target_seg==i,fusion_label==i))
        return np.array(dices)[1:]

    def validate(self, sess, generator, keys=None, summary=False, predict=False, show_tqdm=False):
        if keys is None:
            keys = ['reg_dices_score','seg_dices_score', 'landmark_dist', 'pt_mask', 'jacc_score']
        full_results = dict([(k, list()) for k in keys])
        if not summary:
            full_results['id1'] = []
            full_results['id2'] = []
            if predict:
                full_results['seg1'] = []
                full_results['seg2'] = []
                full_results['imgT1_fixed'] = []
                full_results['imgT1_float'] = []
        tflearn.is_training(False, sess)
        if show_tqdm:
            generator = tqdm(generator)
        i = 0
        for FD in generator:
            i += 1
            
            '''if (i>1):
                break
            '''
            if isinstance(FD, list):
                if 'id1' not in FD[0]:
                    break
                keys.append("warped_seg_moving")
                candidates = []
                for fd in FD:
                    id1 = fd.pop("id1")
                    id2 = fd.pop('id2')
                    #print(id2,id1)
                    results = sess.run(self.get_predictions(
                        *keys), feed_dict=set_tf_keys(fd))
                    candidates.append(np.squeeze(results.pop("warped_seg_moving"))/255.0)
                    
                results["reg_dices"] = np.expand_dims(self.fusion_dices(candidates, np.squeeze(fd['seg1'])),0)
                results["reg_dices_score"] = np.expand_dims(np.mean(results["reg_dices"]),0)
            else:
                fd = FD
                if 'id1' not in fd:
                    break
                id1 = fd.pop('id1')
                id2 = fd.pop('id2')
                #print(id1,id2)
                results = sess.run(self.get_predictions(
                    *keys), feed_dict=set_tf_keys(fd))
            if not summary:
                results['id1'] = id1
                results['id2'] = id2
                if predict:
                    results['seg1'] = fd['seg1']
                    results['seg2'] = fd['seg2']
                    results['imgT1_fixed'] = fd['voxelT1']
                    results['imgT1_float'] = fd['atlasT1']

            mask = np.where([i and j for i, j in zip(id1, id2)])
            for k, v in results.items():
                if k not in full_results:
                    continue
                full_results[k].append(v[mask])
        if 'landmark_dist' in full_results and 'pt_mask' in full_results:
            pt_mask = full_results.pop('pt_mask')
            full_results['landmark_dist'] = [arr * mask for arr,
                                             mask in zip(full_results['landmark_dist'], pt_mask)]
        for k in full_results:
            #print(k)
            #print(np.array(full_results[k]).shape)
            full_results[k] = np.concatenate(full_results[k], axis=0)
            if k == 'reg_dices' or k == 'seg_dices' or k == 'dices1':
                print(k,': ', np.mean(full_results[k], axis=0))
            if summary:
                full_results[k] = full_results[k].mean()
        return full_results

