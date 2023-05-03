import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("/home/ubuntu/bo/CCD/")
import train_ctw.config as cfg

flag = False
	

class EMB_Loss(torch.nn.Module):
	def __init__(self, num_class,alpha=1.0,gamma=2.0,beta=0.):
		#out_constrain is a float, sampling rate.
		super(EMB_Loss, self).__init__()
		self.alpha=alpha
		self.gamma=gamma
		self.num_classes=num_class
		self.out_constrain=beta

	def forward(self, decod_feature, mask):
		depth = decod_feature.shape[1]#b c h w
		emb_loss = 0.
		b_size,_,width,height=mask.shape
		if not self.out_constrain==0.:
			cat_out_list=[]
		for class_index in range(self.num_classes):
			mask_center = mask[:, class_index, :, :]==1.0
			a_class_loss = 0.
			m=(mask_center==True).sum()
			if m==0:
				emb_loss += a_class_loss
				continue
			else:
				mask_center = mask_center.view(b_size, 1, width, height)
				mask_center = mask_center.expand(b_size, depth, width, height)
				x_features = torch.masked_select(decod_feature[:, :, ], mask_center).split(depth)
				n = len(x_features)
				a_loss_emb_min = 1.
				x_1=None
				x_2=None
				for i in range(n):
					for j in range(i + 1, n):
						a_loss_emb = F.cosine_similarity(x_features[i], x_features[j], dim=0).item()
						if a_loss_emb_min>a_loss_emb and self.out_constrain>0.:
							a_loss_emb_min=a_loss_emb
							x_1=x_features[i]
							x_2=x_features[j]


						a_class_loss += a_loss_emb
				if self.out_constrain>0. and not x_1==None:
					cat_out_list.append([x_1,x_2])


				a_class_loss /= (n * (n + 1) / 2)
				emb_loss += a_class_loss
		###for out constrain###
		if self.out_constrain > 0.:
			count_x=0
			out_loss=0.
			for cat_out_index_1 in range(len(cat_out_list)-1):
				for cat_out_index_2 in range(cat_out_index_1+1,len(cat_out_list)):
					for i in range(2):
						xc_1=cat_out_list[cat_out_index_1][i]
						xc_2=cat_out_list[cat_out_index_2][0]
						xc_3=cat_out_list[cat_out_index_2][1]
						loss_emb1 = F.cosine_similarity(xc_1, xc_2, dim=0).item()
						loss_emb2 = F.cosine_similarity(xc_1, xc_3, dim=0).item()
						count_x+=2
						out_loss+=loss_emb1
						out_loss+=loss_emb2
			if count_x==0:
				emb_loss = torch.tensor(emb_loss / (self.num_classes))
				emb_loss = self.alpha * torch.exp(-self.gamma * emb_loss).cuda()
				return emb_loss
			else:
				out_loss/=count_x
				out_loss=torch.tensor(self.out_constrain*out_loss).cuda()
				emb_loss = torch.tensor(emb_loss / (self.num_classes))
				emb_loss = self.alpha * torch.exp(-self.gamma * emb_loss).cuda()
				total_loss=(out_loss+emb_loss)/2
				return total_loss

		else:
			emb_loss = torch.tensor(emb_loss / (self.num_classes))
			emb_loss = self.alpha*torch.exp(-self.gamma * emb_loss).cuda()
			return emb_loss

 
criterion=EMB_Loss(num_class=2,beta=0.0)

def compute_emb_loss(decod_feature,points,beta=0.):
    loss=criterion(decod_feature,points)
    return loss


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SepConv(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        #   depthwise and pointwise convolution, downsample by 2
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)

class smalier_model(nn.Module):
    def __init__(self,):
        super(smalier_model, self).__init__()

        self.auxiliary1 = nn.Sequential(
            SepConv(
                channel_in=32,
                channel_out=64
            ),
            SepConv(
                channel_in=64,
                channel_out=128
            ),
            nn.AdaptiveAvgPool2d((512, 512)),
            SepConv(
                channel_in=128,
                channel_out=256
            ),
            # nn.AdaptiveAvgPool2d((1, 1))
        )

        self.auxiliary2 = nn.Sequential(
            SepConv(
                channel_in=64 ,
                channel_out=128,
            ),
            SepConv(
                channel_in=128,
                channel_out=256,
            ),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.auxiliary3 = nn.Sequential(
            SepConv(
                channel_in=128 ,
                channel_out=256 ,
            ),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.auxiliary4 = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        # for xx in x:
        #     print(xx.shape)
        # print(x[3].shape)
        out1_feature = self.auxiliary1(x[3])#.view(x[3].size(0), -1)
        # out2_feature = self.auxiliary2(x[2]).view(x[3].size(0), -1)
        # out3_feature = self.auxiliary3(x[1]).view(x[3].size(0), -1)
        # out4_feature = self.auxiliary4(x[0]).view(x[3].size(0), -1)

        # feat_list = [out1_feature,out2_feature,out3_feature,out4_feature]
        # for index in range(len(feat_list)):
        #     feat_list[index] = F.normalize(feat_list[index], dim=1)

        # print(out1_feature.shape)
        # # t = F.normalize(out1_feature, dim=1)
        # # print(out1_feature.shape)
        # t=out1_feature
        # print(str(target.shape)+" "+str(t.shape))



        return out1_feature

# if __name__ == '__main__':
#     model = CRAFT().cuda()
#     output = model(torch.randn(1, 3, 512, 512).cuda())
#     print(output.shape)
