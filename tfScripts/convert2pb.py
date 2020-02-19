import tensorflow as tf
from tensorflow.python.tools import freeze_graph 

# network是你自己定义的模型
from lenet import Lenet

# 模型的checkpoint文件地址
ckpt_path = "./output/lenet.ckpt-49"

def main(): 
	tf.reset_default_graph() 

	flow = lenet()
	
    #设置输出类型以及输出的接口名字，为了之后的调用pb的时候使用 
    flow = tf.cast(flow, tf.int8, 'out')

	with tf.Session() as sess:
		#保存图，在./pb_model文件夹中生成model.pb文件
                # model.pb文件将作为input_graph给到接下来的freeze_graph函数
		tf.train.write_graph(sess.graph_def, './pb_model', 'model.pb') 
		#把图和参数结构一起 
		freeze_graph.freeze_graph(
			input_graph='./pb_model/model.pb',
			input_saver='',
			input_binary=False, 
			input_checkpoint=ckpt_path, 
			output_node_names='out',
			restore_op_name='save/restore_all',
			filename_tensor_name='save/Const:0',
			output_graph='./pb_model/frozen_model.pb',
			clear_devices=False,
			initializer_nodes=''
			)

	print("done") 

if __name__ == '__main__': 
	main()