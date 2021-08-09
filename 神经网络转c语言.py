# -*- coding: utf-8 -*-
# @Time : 2021/8/6 10:12
# @Author : Guru
# @QQ:2450967798
# @File : 神经网络转c语言.py

import numpy as np
from keras import models
from keras.models import Model

# 各层神经元数
dense_cell_Num = []


def read_network(file_name):
    model = models.load_model(file_name)
    dense_list = []
    print("开始读取权重...")
    for i in range(1, 20):  # 最大19层 该数字可以加大
        dense = []
        try:
            weight_Dense, bias_Dense = model.get_layer('dense_{}'.format(i)).get_weights()
        except:
            break

        # 遍历每一个圈中的目的是为了四舍五入权重（我使用numpy.around转换失败,所以使用遍历）
        weight_list = [list(item) for item in weight_Dense]
        for index1, y in enumerate(weight_list):
            for index2, x in enumerate(y):
                weight_list[index1][index2] = round(x, 4)
        # print("{}层网络的总共权重为".format(i),len_temp.shape[1]*len_temp.shape[0])
        weight_str = str(weight_list)
        # print(weight_str)
        weight_str = weight_str.replace("[", '{')
        weight_str = weight_str.replace("]", '}')
        # print(weight_str)
        dense.append(weight_str)
        bias_list = [item for item in bias_Dense]
        print("该层偏差数有{}".format(len(bias_list)))
        dense_cell_Num.append(len(bias_list))
        bias_str = str(bias_list)
        bias_str = bias_str.replace("[", '{')
        bias_str = bias_str.replace("]", '}')
        dense.append(bias_str)
        dense_list.append(dense)
        print("*" * 100)
    return dense_list


def test_model(model_name):
    print("开始测试...")
    model = models.load_model(model_name)
    a = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
    y = model.predict(a)
    print("测试结果为：", y)


def copy_weight_to_C(dense_list, act, txtname):
    # 网络层数
    dense_Num = len(dense_list)
    head_code = '''
# include<stdio.h>
# include<math.h>
// 单层神经元个数定义
#define dense0_num {0}
#define input_num {0}'''.format(dense_cell_Num[0])
    for index, cell_Num in enumerate(dense_cell_Num):
        head_code = head_code + '''
#define dense{}_num {}'''.format(index + 1, cell_Num)
    # print(head_code)

    # 先添加第一层网络参数权重
    head_code += '''
float weight_array_1[input_num][dense1_num]={};
float bias_array_1[dense1_num]={};
'''.format(dense_list[0][0], dense_list[0][1])
    # print(head_code)
    for index, weight in enumerate(dense_list):
        if index == 0: continue
        head_code += '''
float weight_array_{}[dense{}_num][dense{}_num]={};
float bias_array_{}[dense{}_num]={};
'''.format(index + 1, index, index + 1, dense_list[index][0],
           index + 1, index + 1, dense_list[index][1])
    # print(head_code)
    head_code += '''
//激活函数
float my_sigmoid(float x)
{
	float e=2.71828,y;
	y=1/(1+pow(e,-x));
	return y;
}
float my_tanh(float x)
{
	float e=2.71828,y;
	y=((pow(e,x)-pow(e,-x))/(pow(e,x)+pow(e,-x)));
	return y;
}
float my_relu(float x)
{
	float y;
	if (x>=0)
		y=x;
	else
		y=0;
	return y;
}
'''  # 添加激活函数
    # print(head_code)
    for item in range(dense_Num):
        head_code += '''
float out{}[dense{}_num] ={{0}};'''.format(item + 1, item + 1)

    for num in range(dense_Num):
        head_code += '''
void desen_{0}(float input_data[dense{2}_num])
{{
	float x,y;
	int i=0,j=0,input=dense{2}_num,out=dense{0}_num;
	//开始计算
	for(i=0;i<input;i++)
	{{
		for(j=0;j<out;j++) 
		{{	
			out{0}[j]=out{0}[j]+input_data[i]*weight_array_{0}[i][j];
			if(i==input-1)
			{{
				//加入偏差
				out{0}[j]=out{0}[j]+bias_array_{0}[j];
				x=out{0}[j];out{0}[j]={3}(x);
			}}
		}}
		input_data[i]=0;	
	}}
}}   
'''.format(num + 1, dense_cell_Num[num], num, "my_" + act[num])
    head_code += '''
    void count(float input_data[input_num])
{{
	out{}[0]=0;
	desen_1(input_data);'''.format(dense_Num)
    for item in range(2, dense_Num + 1):
        head_code += '''
    desen_{}(out{});'''.format(item, item - 1)
    else:
        head_code += "}"
    # print(head_code)
    input_data = ("0.1," * dense_cell_Num[0])[:-1]  # 添加多个值为0.1的验证数据 用来检验网络的结果是否正确
    head_code += '''
int main()
{{	
	
	float input_data[{0}]={{{1}}};
	count(input_data);
	printf("最终结果是这个%lf\\n",out{2}[0]);
	return 0;	
}}'''.format(dense_cell_Num[0], input_data, dense_Num)
    f = open("{}".format(txtname), "a")
    f.write(head_code)
    f.close()
    print("转换完成！！！")


if __name__ == '__main__':
    txtname = '全连接神经网络转换.txt'
    network_name = r'全连接网络.h5'  # 神经网络路径
    dense_list = read_network(network_name)
    act = ['tanh', "relu", "relu", "relu", "relu", "relu", "tanh"]
    # copy_weight_to_C(dense_list, act, txtname)
    test_model(network_name)  # 测试网络转换前后的结果是否相等
    '''
    我暂时没有读取每一层激活函数的方法 因此替换前 将神经网络每一层的激活函数填入act列表 列表的个数等于层数即可
    该文件仅支持全连接神经网络（卷积神经网络暂不支持）
    激活函数 我已经定义了常见的sigmoid tanh relu 可以添加其他的激活函数
    有更好的转换方法，欢迎与我交流 QQ：2450967798
    '''
