import torch
import customDatasetTest
import os
import mymodel

dataset = '../../../datasets/HateSPic/MMHS50K/' # Path to dataset
splits = ['MMHS50K_noOtherHard_lstm_embeddings_test_hate.txt','MMHS50K_noOtherHard_lstm_embeddings_val_hate.txt','MMHS50K_noOtherHard_lstm_embeddings_train_hate.txt']
batch_size = 32
workers = 6
model_name = 'MMHS50K_noOtherHard_Iembeddings_epoch_32_ValAcc_54'

gpus = [0]
gpu = 0
CUDA_VISIBLE_DEVICES=0

output_file_path = dataset + 'img_embeddings/' + model_name + '.txt'
output_file = open(output_file_path, "w")

if os.path.isfile(dataset + '/MMCNN_models/' + model_name + '.pth.tar'):
    state_dict = torch.load(dataset + '/MMCNN_models/' + model_name + '.pth.tar', map_location={'cuda:1':'cuda:0', 'cuda:2':'cuda:0', 'cuda:3':'cuda:0'})
else:
    state_dict = torch.load(dataset + '/MMCNN_models_loss/' + model_name + '.pth.tar', map_location={'cuda:1': 'cuda:0', 'cuda:2': 'cuda:0', 'cuda:3': 'cuda:0'})
    print("no checkpoint found")

model = mymodel.MyModel()
model = torch.nn.DataParallel(model, device_ids=gpus).cuda(gpu)
model.load_state_dict(state_dict)


for split in splits:
    cur_dataset = customDatasetTest.customDatasetTest(dataset, split, Rescale=299)
    loader = torch.utils.data.DataLoader(cur_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, sampler=None)
    print("Computing: " + split)
    with torch.no_grad():
        model.eval()
        for i,(tweet_id, image, image_text, tweet, target) in enumerate(loader):

            image_var = torch.autograd.Variable(image)
            image_text_var = torch.autograd.Variable(image_text)
            tweet_var = torch.autograd.Variable(tweet)

            outputs = model(image_var, image_text_var, tweet_var)

            for idx,el in enumerate(outputs):
                topic_probs_str = ''
                for t in el:
                    topic_probs_str = topic_probs_str + ' ' + str(float(t))
                output_file.write(str(tweet_id[idx]) + ',' + topic_probs_str + '\n')
    print("Done: " + split)

print("DONE")