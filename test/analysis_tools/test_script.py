import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

version = "v1"
test_epochs = range(50,100,2)
out_dir = "./data/%s_output"%(version)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
log_path = out_dir + "/test.log"

for i in test_epochs:
    f = open(log_path, 'a+')  
    print("Begin test epoch %d !"%(i), file=f)
    f.close()
    os.system("python main.py --output_path ./data/%s_output --model_file ../train/checkpoints/%s/epoch_%d.pth"%(version,version,i))
    os.system("python analysis_tools/cal_matrics.py --pred_path .data/%s_output --output_path ./data/%s_output/logs/%d.csv"%(version,version,i))

