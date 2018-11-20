import shutil


wrong_ids_file = '../../../datasets/HateSPic/HateSPic/evaluation_results/LSTM_MMHS10K-v3mm_tweet_text_wrong_ids.txt'
correct_ids_file = '../../../datasets/HateSPic/HateSPic/evaluation_results/MMHSv3mm_SAVED_FCM_I_ADAM_bs32_lrMMe6_lrCNNe7_epoch_118_ValAcc_61_correct_ids.txt'

wrong_ids = []
correct_ids = []
corrected_ids = []

for line in open(wrong_ids_file,'r'):
    data = line.split(',')
    wrong_ids.append(int(data[0]))
# for line in open(correct_ids_file,'r'):
#     data = line.split(',')
#     correct_ids.append([int(data[0]), int(data[1])])
#
# for id in correct_ids:
#     if id[0] in wrong_ids:
#         corrected_ids.append(id)
#
# print("Total of wrong ids: " + str(len(wrong_ids)) + " / Corrected: " + str((len(corrected_ids))))
#
# with open('../../../datasets/HateSPic/HateSPic/wrong_ids.txt','w') as outfile:
#     for id in corrected_ids:
#         outfile.write(str(id[0]) + ',' + str(id[1]) + '\n')


for id in wrong_ids:
    shutil.copyfile('../../../datasets/HateSPic/HateSPic/img_resized/' + str(id) + '.jpg', '../../../datasets/HateSPic/HateSPic/wrong_by_LSTM/' + str(id) + '.jpg')


