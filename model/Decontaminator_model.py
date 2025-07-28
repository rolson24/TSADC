import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch_geometric.nn import GINEConv
import torch_geometric
import scipy
from model.model_dependency import *
from utils.model_utils import *
import math


class TSADC(nn.Module):
    def __init__(
            self,
            input_dim,
            num_nodes,
            dropout,
            num_temporal_layers,
            g_conv,
            num_gnn_layers,
            hidden_dim,
            max_seq_len,
            interval,
            state_dim=64,
            channels=1,
            bidirectional=False,
            temporal_pool="mean",
            prenorm=False,
            postact=None,
            metric="self_attention",
            adj_embed_dim=10,
            gin_mlp=False,
            train_eps=False,
            prune_method="thresh",
            edge_top_perc=0.5,
            thresh=None,
            activation_fn="leaky_relu",
            num_classes=1,
            undirected_graph=True,
            K=3,
            regularizations=["feature_smoothing", "degree", "sparse"],
            residual_weight=0.0,
            masking = 'rm',
            masking_r = 1200,
            masking_r_test = 1200,
            diffuse_T = 200,
            diffuse_beta_0 = 0.0001,
            diffuse_beta_T = 0.02,
            in_channels = 64,
            out_channels = 64,
            num_res_layers = 36,
            res_channels = 256,
            skip_channels = 256,
            diffusion_step_embed_dim_in = 128,
            diffusion_step_embed_dim_mid = 128,
            diffusion_step_embed_dim_out = 128,
            s4_max = 200,
            s4_d_state = 64,
            s4_dropout = 0.0,
            s4_bidirectional = 1,
            s4_layernorm = 1,
            feature_smoothing_weight = 0.05,
            degree_weight = 0.05,
            sparse_weight = 0.05,
            step_in_seq = 3750,
            step_in_seq_test = 3750,
            test_samples = 310,
            lambda_1 = 0.01,
            lambda_2 = 1.2,
            tau = None,
            **kwargs
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.edge_top_perc = edge_top_perc
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.metric = metric
        self.undirected_graph = undirected_graph
        self.K = K
        self.regularizations = regularizations
        self.residual_weight = residual_weight
        self.temporal_pool = temporal_pool
        self.max_seq_len = max_seq_len
        self.interval = interval
        self.prune_method = prune_method
        self.thresh = thresh
        self.masking = masking
        self.masking_r = masking_r
        self.masking_r_test = masking_r_test
        self. diffuse_T = diffuse_T
        self.diffuse_beta_0 = diffuse_beta_0
        self.diffuse_beta_T = diffuse_beta_T
        self.in_channels = in_channels,
        self.out_channels = out_channels,
        self.num_res_layers = num_res_layers,
        self.res_channels = res_channels,
        self.skip_channels = skip_channels,
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in,
        self.diffusion_step_embed_dim_mid = diffusion_step_embed_dim_mid,
        self.diffusion_step_embed_dim_out = diffusion_step_embed_dim_out,
        self.s4_max = s4_max,
        self.s4_d_state = s4_d_state,
        self.s4_dropout = s4_dropout,
        self.s4_bidirectional = s4_bidirectional,
        self.s4_layernorm = s4_layernorm,
        self.feature_smoothing_weight = feature_smoothing_weight,
        self.degree_weight = degree_weight,
        self.sparse_weight = sparse_weight,
        self.step_in_seq = step_in_seq,
        self.step_in_seq_test = step_in_seq_test,
        self.test_samples = test_samples,
        self.lambda_1 = lambda_1,
        self.lambda_2 = lambda_2,
        self.tau = tau,



        self.diffusion_hyperparams = compute_diffusion_hyperparamters(
            self.diffuse_T,
            self.diffuse_beta_0,
            self.diffuse_beta_T,
        )

        # decontaminator
        self.rec_mask_model = Decontaminator(
            in_channels = in_channels,
            res_channels = res_channels,
            skip_channels = skip_channels,
            out_channels = out_channels,
            num_res_layers = num_res_layers,
            diffusion_step_embed_dim_in = diffusion_step_embed_dim_in,
            diffusion_step_embed_dim_mid = diffusion_step_embed_dim_mid,
            diffusion_step_embed_dim_out = diffusion_step_embed_dim_out,
            s4_lmax = s4_max,
            s4_d_state = s4_d_state,
            s4_dropout = s4_dropout,
            s4_bidirectional = s4_bidirectional,
            s4_layernorm = s4_layernorm,
        )



    def forward(self, data, lengths=None, mode = "test"):
        """
        Args:
            data: torch geometric data object
        """
        x_data = data.x # (batch*node, time_seq, 1)
        n_batch = x_data.shape[0] // self.num_nodes
        node = int(x_data.shape[0] / n_batch)

        rec_mask_loss = []
        reg_losses = []
        rec_loss = []
        sampled_x = []
        mask_sampled = []
        logits = []
        pred_x = []
        if mode == "test":
            step_in_seq = self.step_in_seq_test
            masking_r = self.masking_r_test
        else:
            step_in_seq = self.step_in_seq
            masking_r = self.masking_r
       
        for idx in range(0, x_data.shape[1], step_in_seq[0]):
            # decontaminator
            x_tmp = x_data[:, idx:idx+step_in_seq[0],:]
            x_true = x_tmp
            x_tmp = x_tmp.reshape(n_batch, node, x_tmp.shape[1], x_tmp.shape[2])
            x_tmp = x_tmp.reshape(x_tmp.shape[0], x_tmp.shape[1], x_tmp.shape[2] * x_tmp.shape[3])
            x_tmp = x_tmp.permute(0, 2, 1)

            if self.masking == 'rm':
                transposed_mask = mask_RandM(x_tmp[0], masking_r)
            elif self.masking == 'rbm':
                transposed_mask = mask_RandBM(x_tmp[0], masking_r)
            elif self.masking == 'bom':
                transposed_mask = mask_BoM(x_tmp[0], masking_r)

            mask = transposed_mask.permute(1, 0)
            mask = mask.repeat(x_tmp.size()[0], 1, 1).float().cuda()
            not_mask = ~mask.bool()
            batch = x_tmp.permute(0, 2, 1)

            del x_tmp

            assert batch.size() == mask.size() == not_mask.size()

            x_tmp = batch, batch, mask, not_mask

            pred_masked_x, loss_noise = self.rec_mask_loss(
                self.rec_mask_model,
                nn.MSELoss(),
                x_tmp,
                self.diffusion_hyperparams
            )
            if mode == "test":
                num_samples = 20
                sampled_data = self.sampling(self.rec_mask_model, (num_samples, self.num_nodes, batch.shape[2]),
                                           self.diffusion_hyperparams,
                                           cond=batch,
                                           mask=mask)
                sampled_data = sampled_data.reshape(self.num_nodes, batch.shape[2], self.input_dim)
                sampled_x.append(sampled_data)

            del x_tmp

            

            # # linear
            # x = self.linear(x)
            # logits.append(x)
            # loss = nn.MSELoss()
            # rec_ind_loss = loss(x, x_orig)

            if mode == "train":
                rec_mask_loss = loss_noise
                # reg_losses.append(loss_graph)
                # rec_loss.append(rec_ind_loss)
                rec_mask_loss = sum(rec_mask_loss) / len(rec_mask_loss)
                # reg_losses = sum(reg_losses) / len(reg_losses)
                # rec_loss = sum(rec_loss) / len(rec_loss)
                return rec_mask_loss #, reg_losses, rec_loss

        if mode == "test":
            sampled_x = torch.cat(sampled_x, dim=1)
            mask_sampled = torch.cat(mask_sampled, dim=1)

            pred_x = torch.cat(pred_x, dim=1)
            logits = torch.cat(logits, dim=1)
            return (sampled_x, mask_sampled, pred_x, logits)

    def rec_mask_loss(self, rec_mask_net, loss_fn, X, diffusion_hyperparams):

        _dh = diffusion_hyperparams
        T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]
        signal = X[0]
        c = X[1]
        mask = X[2]
        not_mask = X[3]

        B, C, L = signal.shape
        diffusion_steps = torch.randint(T, size=(B, 1, 1)).cuda()
        Alpha_bar = Alpha_bar.cuda()

        #create noise
        z = std_normal(signal.shape)
        z = signal * mask.float() + z * (1 - mask).float()

        # add noise x^t
        transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * signal + torch.sqrt(
            1 - Alpha_bar[diffusion_steps]) * z

        # predict noise e_theta
        epsilon_theta = rec_mask_net(
            (transformed_X, c, mask, diffusion_steps.view(B, 1),))

        # reconstruct x^0
        predicted_X = torch.sqrt(1 / Alpha_bar[diffusion_steps]) * (transformed_X - torch.sqrt(
            1 - Alpha_bar[diffusion_steps]) * epsilon_theta)

        return (predicted_X, loss_fn(epsilon_theta[not_mask], z[not_mask]))


    def sampling(self, net, size, diffusion_hyperparams, cond, mask,  guidance_weight=0):

        _dh = diffusion_hyperparams
        T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
        assert len(Alpha) == T
        assert len(Alpha_bar) == T
        assert len(Sigma) == T
        assert len(size) == 3

        x = std_normal(size)

        with torch.no_grad():
            for t in range(T - 1, -1, -1):
                x = x * (1 - mask).float() + cond * mask.float()
                diffusion_steps = (t * torch.ones((size[0], 1))).cuda()
                epsilon_theta = net((x, cond, mask, diffusion_steps,))
                x = (x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])
                if t > 0:
                    x = x + Sigma[t] * std_normal(size)

        return x



    def regularization_loss(self, x, adj, reduce="mean"):

        batch, num_nodes, _ = x.shape
        n = num_nodes

        loss = {}

        if "feature_smoothing" in self.regularizations:
            curr_loss = feature_smoothing(adj=adj, X=x) / (n ** 2)
            if reduce == "mean":
                loss["feature_smoothing"] = torch.mean(curr_loss)
            elif reduce == "sum":
                loss["feature_smoothing"] = torch.sum(curr_loss)
            else:
                loss["feature_smoothing"] = curr_loss

        if "degree" in self.regularizations:
            ones = torch.ones(batch, num_nodes, 1).to(x.device)
            curr_loss = -(1 / n) * torch.matmul(
                ones.transpose(1, 2), torch.log(torch.matmul(adj, ones))
            ).squeeze(-1).squeeze(-1)
            if reduce == "mean":
                loss["degree"] = torch.mean(curr_loss)
            elif reduce == "sum":
                loss["degree"] = torch.sum(curr_loss)
            else:
                loss["degree"] = curr_loss

        if "sparse" in self.regularizations:
            curr_loss = (
                    1 / (n ** 2) * torch.pow(torch.norm(adj, p="fro", dim=(-1, -2)), 2)
            )

            if reduce == "mean":
                loss["sparse"] = torch.mean(curr_loss)
            elif reduce == "sum":
                loss["sparse"] = torch.sum(curr_loss)
            else:
                loss["sparse"] = curr_loss

        if "symmetric" in self.regularizations and self.undirected_graph:
            curr_loss = torch.norm(adj - adj.transpose(1, 2), p="fro", dim=(-1, -2))
            if reduce == "mean":
                loss["symmetric"] = torch.mean(curr_loss)
            elif reduce == "sum":
                loss["symmetric"] = torch.sum(curr_loss)
            else:
                loss["symmetric"] = curr_loss

        return loss

    def aggregate_regularization_losses(self, reg_loss_dict):
        reg_loss = 0.0
        for k in self.args.regularizations:
            if k == "feature_smoothing":
                reg_loss = (reg_loss + self.args.feature_smoothing_weight * reg_loss_dict[k])
            elif k == "degree":
                reg_loss = reg_loss + self.args.degree_weight * reg_loss_dict[k]
            elif k == "sparse":
                reg_loss = reg_loss + self.args.sparse_weight * reg_loss_dict[k]
            else:
                raise NotImplementedError()
        return reg_loss


