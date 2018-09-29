import tensorflow as tf
import numpy as np
import os
import random
import PIL.Image as Image
import cv2
import os
import copy
import time
import P3D
import DataGenerator
from settings import *

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--txt',type=str)
parser.add_argument('--model',type=str)

args=parser.parse_args()

IS_DA=True  #true if using data augmentation

def average_gradients(tower_grads):
	average_grads = []
	for grad_and_vars in zip(*tower_grads):
		grads = []
		for g, v in grad_and_vars:
			if g is not None:
				expanded_g = tf.expand_dims(g, 0)
				grads.append(expanded_g)
		if grads:
			grad = tf.concat(grads, 0)
			grad = tf.reduce_mean(grad, 0)
		else:
			grad = None
		v = grad_and_vars[0][1]
		grad_and_var = (grad, v)
		average_grads.append(grad_and_var)
	return average_grads

def compute_loss(name_scope,logit,labels):
    cross_entropy_mean=tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logit))
    tf.summary.scalar(name_scope+'_cross_entropy',
                     cross_entropy_mean
                     )
    weight_decay_loss=tf.get_collection('weightdecay_losses')
    tf.summary.scalar(name_scope+'_weight_decay_loss',tf.reduce_mean(weight_decay_loss))
    total_loss=cross_entropy_mean+weight_decay_loss
    tf.summary.scalar(name_scope+'_total_loss',tf.reduce_mean(total_loss))
    return total_loss

def compute_accuracy(logit,labels):
    correct=tf.equal(tf.argmax(logit,1),labels)
    acc=tf.reduce_mean(tf.cast(correct,tf.float32))
    return acc

def placeholder_inputs(batch_size):
  	images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         	NUM_FRAMES_PER_CLIP,
                                                         	CROP_SIZE,
                                                         	CROP_SIZE,
                                                         	RGB_CHANNEL))
  	labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
  	return images_placeholder, labels_placeholder

def run():
	MOVING_AVERAGE_DECAY=0.9
	MODEL_PATH=''
	USE_PRETRAIN=False
	MAX_STEPS=2000

	dataloader=DataGenerator.DataGenerator(filename=args.txt,
                                batch_size=BATCH_SIZE * GPU_NUM,
                                num_frames_per_clip=NUM_FRAMES_PER_CLIP,
                                shuffle=True,is_da=IS_DA)
	

	with tf.Graph().as_default():

		global_step=tf.get_variable('global_step',[],initializer=tf.constant_initializer(0),trainable=False)
		images_placeholder, labels_placeholder = placeholder_inputs(BATCH_SIZE * GPU_NUM)
		
		#if necessary,you can set different learning rate for FC layers and the other individually.
		learning_rate_stable = tf.train.exponential_decay(0.0005,
		                                           global_step,decay_steps=2100,decay_rate=0.6,staircase=True)
		learning_rate_finetune = tf.train.exponential_decay(0.0005,
		                                           global_step,decay_steps=2100,decay_rate=0.6,staircase=True)
		
		opt_stable=tf.train.AdamOptimizer(learning_rate_stable)
		opt_finetuning=tf.train.AdamOptimizer(learning_rate_finetune)

		tower_grads1 = []
		tower_grads2 = []
		logits = []

		for gpu_index in range(0, GPU_NUM):
			print("gpu_index:", gpu_index)
			with tf.variable_scope('p3d', reuse=bool(gpu_index != 0)):
				with tf.device('/gpu:%d' % gpu_index):
					logit=P3D.inference_p3d(
							images_placeholder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE,:,:,:,:],
							0.5,
							BATCH_SIZE)
					loss_name_scope = ('gpud_%d_loss' % gpu_index)
					loss = compute_loss(
					    			loss_name_scope,
					    			logit,
					    			labels_placeholder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE])
						
					# use last tower statistics to update the moving mean/variance 
					batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

					# Reuse variables for the next tower.
					# tf.get_variable_scope().reuse_variables()

					varlist1=[]
					varlist2=[]
					for param in tf.trainable_variables():
						if param.name!='p3d/fc/bias:0' and param.name!='p3d/fc/kernel:0':
						    varlist1.append(param)
						else:
						    varlist2.append(param)

					print("param_size: ",np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

					grads1 = opt_stable.compute_gradients(loss, varlist1)
					grads2 = opt_finetuning.compute_gradients(loss, varlist2)

					tower_grads1.append(grads1)
					tower_grads2.append(grads2)
					logits.append(logit)

		logits = tf.concat(logits,0)
		acc=compute_accuracy(logits,labels_placeholder)
		tf.summary.scalar('accuracy',acc)
		
		grads1 = average_gradients(tower_grads1)
		grads2 = average_gradients(tower_grads2)

		# when using BN,this dependecy must be built.
		# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		# batchnorm_updates_op = tf.group(*batchnorm_updates)
		apply_gradient_op1 = opt_stable.apply_gradients(grads1)
		apply_gradient_op2 = opt_finetuning.apply_gradients(grads2, global_step=global_step)
		
		with tf.control_dependencies(batchnorm_updates):
		    optim_op_group=tf.group(apply_gradient_op1,apply_gradient_op2)
		


		variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,num_updates=global_step)
		variable_averages_op=variable_averages.apply(tf.trainable_variables())
		train_op=tf.group(optim_op_group,variable_averages_op)
		
		#when using BN,only store trainable parameters is not enough,cause MEAN and VARIANCE for BN is not
		#trainable but necessary for test stage.
		saver=tf.train.Saver(tf.global_variables())
		init=tf.global_variables_initializer()
		sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		sess.run(init)
		if USE_PRETRAIN:
		    saver.restore(sess,MODEL_PATH)
		    print('Checkpoint reloaded.')
		else:
		    print('Training from scratch.')
		merged=tf.summary.merge_all()
		train_writer=tf.summary.FileWriter('./visual_logs/'+args.model+'/train',sess.graph)
		test_writer=tf.summary.FileWriter('./visual_logs/'+args.model+'/test',sess.graph)
		duration=0
		print('Start training.')
		for step in range(1,MAX_STEPS):
		    sess.graph.finalize()
		    start_time=time.time()
		    train_images,train_labels,_=dataloader.next_batch()
		    sess.run(train_op,feed_dict={
		                    images_placeholder:train_images,
		                    labels_placeholder:train_labels})
		    duration+=time.time()-start_time
		    
		    
		    if step!=0 and step % 10==0:
		        curacc,curloss=sess.run([acc,loss],feed_dict={
		                    images_placeholder:train_images,
		                    labels_placeholder:train_labels})
		        print('Step %d: %.2f sec -->loss : %.4f =====acc : %.2f' % (step, duration,np.mean(curloss),curacc))
		        duration=0
		    if step!=0 and step % 50==0:
		        mer=sess.run(merged,feed_dict={
		                    images_placeholder:train_images,
		                    labels_placeholder:train_labels})
		        train_writer.add_summary(mer, step)
		    if step >7000 and step % 800==0 or (step+1)==MAX_STEPS:
		        pass
		        #saver.save(sess,'./TFCHKP_{}'.format(step),global_step=step)
		    
		print('done')   

if __name__=='__main__':
	print('Preparing for training,this may take several seconds.')
	run()        
        
    


