import torch
import customDatasetTest
import os
import mymodel

dataset = '../../../datasets/HateSPic/HateSPic/' # Path to dataset
split = 'lstm_embeddings_test_hate.txt'
batch_size = 12
workers = 6
model_name = 'HateSPic_inceptionv3_bs32_decay50_all_epoch_88_best'

gpus = [0]
gpu = 0

if not os.path.exists(dataset + 'results/' + model_name):
    os.makedirs(dataset + 'results/' + model_name)

output_file_path = dataset + 'results/' + model_name + '/test.txt'
output_file = open(output_file_path, "w")

if os.path.isfile(dataset + '/models/' + model_name + '.pth.tar'):
    state_dict = torch.load(dataset + '/models/' + model_name + '.pth.tar')
else:
    print("no checkpoint found")


model = mymodel.MyModel()
model = torch.nn.DataParallel(model, device_ids=gpus).cuda(gpu)

model.load_state_dict(state_dict)

test_dataset = customDatasetTest.customDatasetTest(dataset, split, Rescale=299)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, sampler=None)

with torch.no_grad():
    for i,(tweet_id, image, image_text, tweet, target) in enumerate(test_loader):
        outputs = model(image, image_text, tweet)
        for idx,el in enumerate(outputs):
            topic_probs_str = ''
            for t in el:
                topic_probs_str = topic_probs_str + ',' + str(float(t))
            output_file.write(str(tweet_id[idx]) + ',' + str(int(target[idx])) + topic_probs_str + '\n')

        print(str(i) + ' / ' + str(len(test_loader)))