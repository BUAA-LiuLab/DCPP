import matplotlib.pyplot as plt
import numpy as np

flag = r'Curcumin'
path = r'C:\Users\10946\Desktop\paper_read\AI_molecular_predict_code\GraphDPA-main\GraphDPA-main\example_saved_data\results\\' + flag + '.txt'

with open(path, 'rb')as f:
    lines = f.read().decode(encoding='utf-8').split('\r\n')

list_pathway = []
list_score = []

for line in lines[1:]:
    if line:
        line_list = line.split('\t')
        list_pathway.append([line_list[1],float(line_list[-1])])

list_pathway.sort(key=lambda x: x[1],reverse=True)
print(list_pathway)

label = [i[0][8:] for i in list_pathway[:20]]
score = [i[1] for i in list_pathway[:20]]
for i in label:
    print(i)
x = np.arange(len(label))  # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots(figsize=[8,5])
# plt.figure(figsize=[8,6])
rects1 = ax.bar(x, score, width, color='#BEBEBE')

list_index_red = [5,6,10,13,15,17,20]
list_index_red = [i-1 for i in list_index_red]
x_red = [x[i] for i in list_index_red]
score_red = [score[i] for i in list_index_red]
rects2 = ax.bar(x_red, score_red, width, color='#FF0000')
# rects2 = ax.bar(x + width/2, women_means, width, label='Women')
plt.ylim(0.6, 0.9)
y = [0.6,0.7,0.8,0.9]
plt.yticks(y)
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title(flag)
ax.set_xticks(x)
plt.xticks(x, x, rotation=90)
plt.yticks(y,y,rotation=90)
ax.yaxis.tick_right()
ax.set_xticklabels(label)
plt.savefig(r'C:\Users\10946\Desktop\paper_read\AI_molecular_predict_code\GraphDPA-main\GraphDPA-main\example_saved_data\results\\'+flag+'.svg',format='svg')
# ax.legend()
plt.show()
