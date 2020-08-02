import numpy

import network1


def score_net(net, test_data_list):
    score_card = []
    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        # print(correct_label, 'Входной символ')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs=net.query(inputs)
        label=numpy.argmax(outputs)
        # print(label,'Ответ')
        if correct_label == label:
            score_card.append(1)
        else:
            score_card.append(0)

    return score_card

# количество входных, скрытых и выходных узлов
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

# коэффициент обучения
learning_rate = 0.2
net1 = network1.Network1(input_nodes, hidden_nodes, output_nodes, learning_rate)

train_file = open('mnist_dataset/mnist_train_100.csv', 'r')
# 60 000 записей для обучения
# train_file = open('/home/vasi/temp/mnist_train.csv', 'r')
train_list = train_file.readlines()
train_file.close()

epochs=3
for e in range(epochs):
    # тренировка нейронной сети
    # перебрать все записи в тренировочном наборе данных
    for record in train_list:
        all_values = record.split(',')
        # масштабировать и сместить входные значения
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # создать целевые выходные значения (все равны 0,01, за исключением
        # желаемого маркерного значения, равного 0,99)
        targets = numpy.zeros(output_nodes) + 0.01

        # all_values[0] - целевое маркерное значение для данной записи
        targets[int(all_values[0])] = 0.99
        net1.train(inputs, targets)

# загрузить в список тестовый набор данных CSV-файла набора MNIST
test_data_file = open('mnist_dataset/mnist_test_10.csv')
# 10 000 записей для проверки
# test_data_file = open('/home/vasi/temp/mnist_test.csv')

test_data_list = test_data_file.readlines()
test_data_file.close()

all_values = test_data_list[0].split(',')
# print(all_values[0])
inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
result = net1.query(inputs)
# print(result)

score=score_net(net1, test_data_list)
# print(score)
score_array = numpy.asarray(score)
print(score_array.sum()/score_array.size)
# print(targets)
# print(all_values[0])
# print(targets[4])
#
# image_array=numpy.asfarray(all_values[1:]).reshape((28,28))
# im=plt.imshow(image_array, cmap='Greys',interpolation='None')
# plt.show()

# for i in range(10):
#     print(all_values[i])
#     image_array=numpy.asfarray(all_values[1:]).reshape((28,28))
#     im=plt.imshow(image_array, cmap='Greys',interpolation='None')
#     plt.show()
