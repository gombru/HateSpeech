files = ['50k_worker_1','5k_worker2','5k_worker3','45k_worker2_worker3']
out_file = open("../../../datasets/HateSPic/AMT/MMHS2/results/50k_3workers.csv",'w')
count = 0
for file in files:
    count_file = 0
    for i,line in enumerate(open("../../../datasets/HateSPic/AMT/MMHS2/results/" + file + ".csv")):
        if i == 0: continue
        count += 1
        count_file+=1
        out_file.write(line+'\n')
    print(count_file)
print(count)