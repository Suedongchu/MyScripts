import tensorflow as tf
import tensorflow.contrib.slim as slim
class Lenet:
	mean = 0
	sigma = 0.1
	
	def __init__(self):
		self._build_graph()
	
	def _build_graph(self,network_name='Lenet'):
		self._setup_placeholders_graph()
		self._build_network_graph(network_name)
		self._compute_loss_graph()
		self._compute_acc_graph()
		self._create_train_op_graph()
		self.merged_summary = tf.summary.merge_all()
		return self.logits	

	def _setup_placeholders_graph(self):
		self.x = tf.placeholder("float", shape=[None, 32, 32 , 1],name='x')
		self.y_ = tf.placeholder("float", shape=[None , 10 ], name='y_')

	def _image_summary_conv(self,conv_W,filter_shape,conv):
		with tf.name_scope('visual') as v_s:
			x_min = tf.reduce_min(conv_W)
			x_max = tf.reduce_max(conv_W)
			kernel_0_to_1 = (conv_W - x_min) / (x_max - x_min)
			kernel_transposed = tf.transpose(kernel_0_to_1, [3, 2, 0, 1])
			conv_W_img = tf.reshape(kernel_transposed, [-1, filter_shape[0], filter_shape[1], 1])
			tf.summary.image('conv_w', conv_W_img, max_outputs=filter_shape[3])
			feature_img = conv[0:1, :, :, 0:filter_shape[3]]
			feature_img = tf.transpose(feature_img, perm=[3, 1, 2, 0])
			tf.summary.image('feature', feature_img, max_outputs=filter_shape[3])

	def _image_summary_fc(self,fc_W,filter_shape):
		with tf.name_scope('visual') as v_s:
			x_min = tf.reduce_min(fc_W)
			x_max = tf.reduce_max(fc_W)
			kernel_0_to_1 = (fc_W - x_min) / (x_max - x_min)
			fc_W_img = tf.reshape(kernel_0_to_1, [-1, filter_shape[0], filter_shape[1], 1])
			tf.summary.image('fc_w', fc_W_img, max_outputs=1)


	def _cnn_layer(self, x, filter_shape , conv_strides , padding_tag='VALID',scope_name=None , W_name=None , b_name=None):
		with tf.name_scope(scope_name) as scope:
			conv_W = tf.Variable(tf.truncated_normal(shape = filter_shape, mean = self.mean, stddev = self.sigma) , name = W_name)
			conv_b = tf.Variable(tf.zeros(filter_shape[3]),name = b_name)
			conv_result = tf.nn.conv2d(x, conv_W, conv_strides, padding = padding_tag) + conv_b
			act = tf.nn.relu(conv_result)
			tf.summary.histogram("weights", conv_W)
			tf.summary.histogram("biases", conv_b)
			tf.summary.histogram("activations", act)
			self._image_summary_conv(conv_W = conv_W,filter_shape = filter_shape,conv = act)
			return act

	def _pooling_layer(self , scope_name , x, pool_ksize , pool_strides , padding_tag='VALID'):
		with tf.name_scope(scope_name) as scope:
			return tf.nn.max_pool(x, ksize = pool_ksize, strides = pool_strides,padding = padding_tag)

	def _fully_connected_layer(self , scope_name , W_name , b_name , x , W_shape):
		with tf.name_scope(scope_name) as scope:
			fc_W = tf.Variable(tf.truncated_normal(shape=W_shape, mean=self.mean, stddev=self.sigma),name=W_name)
			fc_b = tf.Variable(tf.zeros(W_shape[1]),name=b_name)
			fc_result = tf.matmul(x, fc_W) + fc_b
			act = tf.nn.relu(fc_result)
			tf.summary.histogram("weights", fc_W)
			tf.summary.histogram("biases", fc_b)
			tf.summary.histogram("activations", act)
			self._image_summary_fc(fc_W=fc_W,filter_shape=W_shape)
			return act

	def _build_network_graph(self , scope_name):
		with tf.name_scope(scope_name):
			tf.summary.image('input', self.x, 3)

			# SOLUTION: Layer 1: Convolutional. Input = 32x32x1*n. Output = 28x28x6*n.
			conv1 = self._cnn_layer(scope_name = "Layer1", W_name = "conv1_W" , b_name = "conv1_b" , x = self.x, filter_shape = (5, 5, 1, 6) , conv_strides = [1, 1, 1, 1])

			# SOLUTION: Pooling. Input = 28x28x6*n. Output = 14x14x6*n.
			conv1 = self._pooling_layer(scope_name = "Layer2" , x = conv1 , pool_ksize = [1, 2, 2, 1] , pool_strides =[1, 2, 2, 1])

			# SOLUTION: Layer 2: Convolutional. Output = 10x10x16*n.
			conv2 = self._cnn_layer(scope_name = "Layer3" , W_name = "conv2_W" , b_name = "conv2_b" , x = conv1 , filter_shape = (5, 5, 6, 16) , conv_strides = [1, 1, 1, 1])

			# SOLUTION: Pooling. Input = 10x10x16*n. Output = 5x5x16*n.
			conv2 = self._pooling_layer(scope_name="Layer4", x=conv2, pool_ksize=[1, 2, 2, 1], pool_strides=[1, 2, 2, 1])

			# SOLUTION: Flatten. Input = 5x5x16*n. Output = 400*n.
			fc0 = tf.reshape(conv2,shape = (-1 , 400))

			# SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
			fc1 = self._fully_connected_layer(scope_name="Layer5", W_name = "fc1_W", b_name = "fc1_b", x = fc0, W_shape = (400, 120))

			# SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
			self.fc2 = self._fully_connected_layer(scope_name="Layer6", W_name="fc2_W", b_name="fc2_b", x=fc1,W_shape=(120, 84))

			# SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
			self.logits = self._fully_connected_layer(scope_name="Layer7", W_name="fc3_W", b_name="fc3_b", x=self.fc2,W_shape=(84, 10))
			tf.summary.histogram("y_predicted", self.logits)

	def _compute_loss_graph(self):
		self.loss = slim.losses.softmax_cross_entropy(self.logits, self.y_)
		tf.summary.scalar("cross_entropy", self.loss)

	def _compute_acc_graph(self):
		correct_prediction = tf.equal(tf.argmax(self.logits,1),tf.argmax(self.y_ ,1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction ,'float'))
		tf.summary.scalar("accuracy", self.accuracy)

	def _create_train_op_graph(self):
		self.train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(self.loss)