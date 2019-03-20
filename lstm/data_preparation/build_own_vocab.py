# # Load glove.twitter.27B.200d vocab and add words for tweets ids corresponding to image features
model_name = 'MMHSv3mm_ImageEmbeddings200_ALL_ADAM_bs32_lrMMe6_lrCNNe7_epoch_160_ValAcc_62_ValLoss_0.65'
dataset = '../../../../datasets/HateSPic/HateSPic/' # Path to dataset

# image_features_path = dataset + 'img_embeddings/' + model_name + '.txt'
# with open('../.vector_cache/mmhsv3_glove.twitter.27B.200d.txt', 'w') as file:
#     for line in open(image_features_path,'r'):
#         # data = line.split(',')
#         # new_line = ""
#         # for el in data:
#         #     new_line += el
#         new_line = line
#         new_line = new_line.replace('\n', '').replace('\r', '').replace(',', '')
#         file.write(new_line + '\n')

# print("DONE")
# c = 0
# with open('../.vector_cache/mmhsv3_glove.twitter.27B.200d.txt', 'r') as file:
#     for line in file:
#         c += 1
#         # print line
# print("Num lines = " + str(c))


with open('../.vector_cache/glove.twitter.27B.200d.txt.pt', 'r') as file:
    for line in file:
        print line


print("DONE")