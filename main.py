
import random
import time
import scipy.io as sio
from torch.utils.data import DataLoader
from utils import plots
import argparse
from utils.utils import *
from utils.data_read import data_get_func
from models.FACM import FACM


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def seed_torch(seed=1):
    '''
    Keep the seed fixed thus the results can keep stable
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ---------------defined paramaters------------------ #
# # Setting Params
parser = argparse.ArgumentParser(description='Training for HSI Unmixing')
parser.add_argument('-d', '--dataset', dest='dataset', choices=['S1', 'JR', 'UR', 'UR5', 'UR6', 'SA', 'AP'],
                    default='JR', help="Name of dataset.")
parser.add_argument('-i', '--iter', type=int, dest='iter', default=1, help="No of Monte Carlo test")
parser.add_argument('-e', '--epochs', type=int, default=450, help='epoch number')
parser.add_argument('-ds', '--down_ratio', type=int, default=16, help='down ratio')
parser.add_argument('-nq', '--num_queries', type=int, default=50, help='query number')
parser.add_argument('-b', '--batch_size', type=int, default=1, help='number of batch size')
parser.add_argument('-n', '--num_copies', type=int, default=1, help='number of copies')
parser.add_argument('--model_name', type=str, default='UNMamba', help='model used')
parser.add_argument('--seed', type=int, default=11, help='number of seed')

parser.add_argument('-l', '--lr', type=float, default=5e-3, help='learning rate')
parser.add_argument('--weight_mse', type=float, default=2, help='weight_mse')
parser.add_argument('--weight_sad', type=float, default=1, help='weight_sad')
parser.add_argument('--weight_sid', type=float, default=0, help='weight_sid')
parser.add_argument('--weight_endm', type=float, default=1, help='weight_endm')
parser.add_argument('--weight_aban', type=float, default=0.02, help='weight_endm')

parser.add_argument('--dropout', type=float, default=1e-2, help='weight_endm')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
args = parser.parse_args()


seed_torch(seed=args.seed)
# define device
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# create saving path
dataset_query = {"AP": "Apex", "JR": "Jasper Ridge", "SA": "Samson", "UR": "Urban", "UR5": "Urban5", "UR6": "Urban6"}
workspace = os.path.abspath(".")
file_path = os.path.join(workspace, 'dataset', dataset_query[args.dataset])
save_path = os.path.join('results', args.model_name, dataset_query[args.dataset])

if not os.path.exists(save_path):
    os.makedirs(save_path)

model_save_path = os.path.join(save_path, 'model')
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

fig_save_path = os.path.join(save_path, 'index_fig')
if not os.path.exists(fig_save_path):
    os.makedirs(fig_save_path)

# 将预设参数保存为txt文件
with open(os.path.join(save_path, 'argparser_params.txt'), 'w') as f:
    for arg in vars(args):
        f.write(f"{arg}: {getattr(args, arg)}\n")
start_time=time.time()
data_get = data_get_func(dataname=dataset_query[args.dataset], file_path=file_path)
data_hsi_img = data_get.data_hsi.reshape(data_get.num_bands, data_get.num_rows, data_get.num_cols).transpose((0, 2, 1))
data_aban_img = data_get.data_aban.reshape(data_get.num_endm, data_get.num_rows, data_get.num_cols).transpose((0, 2, 1))
print(data_hsi_img.shape)

# shape: 1x198x100x100
torch_hsi = torch.from_numpy(data_hsi_img).to(torch.float32).to(device).unsqueeze(0).repeat(args.num_copies, 1, 1, 1)
torch_aban = torch.from_numpy(data_aban_img).to(torch.float32).to(device)

train_data_loader = DataLoader(torch_hsi, batch_size=args.batch_size)

model = FACM(
    height=data_hsi_img.shape[1],
    width=data_hsi_img.shape[2],
    num_band=data_get.num_bands, d_model=64,
    num_endm=data_get.num_endm, num_queries_times=args.num_queries, ds=args.down_ratio,
    dropout=args.dropout
                      ).to(device)

my_loss = My_Loss(
            num_bands=data_get.num_bands,
            weight_mse=args.weight_mse,
                  weight_sad=args.weight_sad,
                  weight_endm=args.weight_endm,
                  weight_fft=1e-5)

data_get = data_get_func(dataname=dataset_query[args.dataset], file_path=file_path)
data_hsi_img = data_get.data_hsi.reshape(data_get.num_bands, data_get.num_rows, data_get.num_cols).transpose((0, 2, 1))
data_aban_img = data_get.data_aban.reshape(data_get.num_endm, data_get.num_rows, data_get.num_cols).transpose((0, 2, 1))
torch_hsi = torch.from_numpy(data_hsi_img).to(torch.float32).to(device).unsqueeze(0).repeat(args.num_copies, 1, 1, 1)
torch_aban = torch.from_numpy(data_aban_img).to(torch.float32).to(device)
hsi_mean_tensor = torch.from_numpy(data_get.get_hsi_mean()).to(torch.float32).to(device)

train_data_loader = DataLoader(torch_hsi, batch_size=args.batch_size, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=0.9)

model.train()

for epoch in range(0, args.epochs):
    loss_print = 0
    loss_sad_print = 0
    loss_mse_print = 0
    loss_endm_print = 0
    loss_aban_print = 0
    loss_fft_print = 0
    temp_i = 0
    for i, x in enumerate(train_data_loader):
        temp_i += 1
        pred_linear, pred_abun, pred_endm = model(x)
        loss_sad, loss_mse, loss_endm, loss_aban, loss_fft, loss = my_loss(x, pred_linear, pred_endm, hsi_mean_tensor, pred_aban=pred_abun)
        optimizer.zero_grad()
        loss.backward()
        for p in model.query_embed.weight:
            p.data.clamp_(1e-7, 1)
        optimizer.step()
        loss_sad_print += loss_sad
        loss_mse_print += loss_mse
        loss_endm_print += loss_endm
        loss_aban_print += loss_aban
        loss_fft_print += loss_fft
        loss_print += loss
    loss_print = loss_print / (temp_i + 1)
    loss_sad_print = loss_sad_print / (temp_i + 1)
    loss_mse_print = loss_mse_print / (temp_i + 1)
    loss_endm_print = loss_endm_print / (temp_i + 1)
    loss_aban_print = loss_aban_print / (temp_i + 1)
    loss_fft_print = loss_fft_print / (temp_i + 1)

    if (epoch+1) % 100 == 0:
        print("Epoch: %d/%d" % (epoch + 1, args.epochs),
              "| lr: %5f" % (optimizer.param_groups[0]['lr']),
              "| loss_sad : %.4f" % loss_sad_print.cpu().data.numpy(),
              "| loss_mse : %.4f" % loss_mse_print.cpu().data.numpy(),
              "| loss_endm: %.4f" % loss_endm_print.cpu().data.numpy(),
              "| loss_aban: %.4f" % loss_aban_print.cpu().data.numpy(),
              "| loss_fft : %.8f" % loss_fft_print.cpu().data.numpy(),
              "| loss: %.4f" % loss_print.cpu().data.numpy()
              )
    scheduler.step()
print('运行时间：',time.time()-start_time)
model.eval()
torch.save(model.state_dict(), os.path.join(model_save_path, 'model.pth'))
test_hsi_img = data_hsi_img.copy()
test_aban_img = data_aban_img.copy().transpose((2, 1, 0))
test_endm = data_get.data_endm.copy()

torch_test_hsi = torch.from_numpy(test_hsi_img).to(torch.float32).to(device).unsqueeze(0)
re_linear, abu_est, endm_ = model(torch_test_hsi)

abu_est = abu_est.squeeze(0).permute(2, 1, 0).detach().cpu().numpy()
re_result = re_linear.squeeze(0).permute(2, 1, 0).detach().cpu().numpy()
est_endmem = model.get_endmember().detach().cpu().numpy()
est_endmem = est_endmem.T

# index = [3, 1, 2, 0] m * 2
# index = [3, 0, 2, 1] m * 1
# index = [2, 0, 3, 1] # FFT 1
# index = [3, 1, 2, 0] # FFT 2


rmse_cls, mean_rmse, index = compute_rmse_with_best_matching(test_aban_img, abu_est)
print("Class-wise RMSE value:")
for i in range(data_get.num_endm):
    print("Class", i + 1, ":", rmse_cls[i])
print("Mean RMSE:", mean_rmse)

sad_cls, mean_sad, _ = compute_sad_with_best_matching(est_endmem, test_endm)
print("Class-wise SAD value:")
for i in range(data_get.num_endm):
    print("Class", i + 1, ":", sad_cls[i])
print("Mean SAD:", mean_sad)

print("RMSE结果：")
for i in range(data_get.num_endm):
    print(f"{rmse_cls[i]:.6f}")
print(f"{mean_rmse:.6f}")

print("\nSAD结果：")
for i in range(data_get.num_endm):
    print(f"{sad_cls[i]:.6f}")
print(f"{mean_sad:.6f}")

output_path = './Results'
method_name = 'FACM'
mat_folder = output_path + '/' + method_name + '/Jasper/' + 'mat'
if not os.path.exists(mat_folder):
    os.makedirs(mat_folder)

abu_est[:, :, np.arange(data_get.num_endm)] = abu_est[:, :, index]
est_endmem[:, np.arange(data_get.num_endm)] = est_endmem[:, index]

sio.savemat(mat_folder + '/' + method_name + '.mat', {'A': abu_est,
                                                                          'E': est_endmem})

plots.plot_abundance(test_aban_img.transpose((1, 0, 2)), abu_est.transpose((1, 0, 2)),
                     data_get.num_endm,
                     save_dir=fig_save_path)
plots.plot_endmembers(data_get.data_endm, est_endmem, data_get.num_endm,
                      save_dir=fig_save_path)
