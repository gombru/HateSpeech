import shutil

base_path = '../../../datasets/HateSPic/MMHS50K/evaluation_results/'

# wrong_ids_file = 'MMHS50K_noOther_FCM_TT_ADAM_bs32_lrMMe6_lrCNNe7_epoch_4_ValAcc_65_wrong_ids.txt'
# correct_ids_file = 'MMHS50K_noOther_FCM_I_ADAM_bs32_lrMMe6_lrCNNe7_epoch_40_ValAcc_55_correct_ids.txt'

wrong_ids_file = 'MMHS50K_noOtherHard_FCM_I_epoch_97_ValAcc_55_correct_ids.txt'
correct_ids_file = 'MMHS50K_noOtherHard_FCM_ALL_epoch_105_ValAcc_65_correct_ids.txt'

wrong_ids = []
correct_ids = []
corrected_ids = []

for line in open(base_path + wrong_ids_file,'r'):
    data = line.split(',')
    wrong_ids.append(int(data[0]))

for line in open(base_path + correct_ids_file,'r'):
    data = line.split(',')
    correct_ids.append([int(data[0]), int(data[1])])

for id in correct_ids:
    if id[0] in wrong_ids:
        corrected_ids.append(id[0])

print("Toral of correct ids: " + str(len(correct_ids)))
print("Total of wrong ids: " + str(len(wrong_ids)) + " / Corrected: " + str((len(corrected_ids))))
#
# with open('../../../datasets/HateSPic/MMHS50K/wrong_ids.txt','w') as outfile:
#     for id in corrected_ids:
#         outfile.write(str(id[0]) + ',' + str(id[1]) + '\n')
#

# for id in corrected_ids:
#     shutil.copyfile('../../../datasets/HateSPic/MMHS50K/img_resized/' + str(id) + '.jpg', '../../../datasets/HateSPic/MMHS50K/corrected_by_I/' + str(id) + '.jpg')


