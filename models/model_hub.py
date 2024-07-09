from .mtmt import *

def mtmt_res_emb_v0():
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=38), treat_feat_enc=nn.Embedding(num_embeddings=10, embedding_dim=16), task_names=['label_nextday_login'],
                 t_dim=1, u_dim=128, tu_dim=256)

def mtmt_res_emb_v0_1():
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=38), treat_feat_enc=nn.Embedding(num_embeddings=10, embedding_dim=128), task_names=['label_nextday_login'],
                 t_dim=1, u_dim=128, tu_dim=256)

def mtmt_res_emb_v0_2():
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=38), treat_feat_enc=nn.Embedding(num_embeddings=10, embedding_dim=16), task_names=['label_nextday_login'],
                 t_dim=1, u_dim=128, tu_dim=128)

def mtmt_res_emb_v0_3():
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=38), treat_feat_enc=nn.Embedding(num_embeddings=10, embedding_dim=16), task_names=['label_nextday_login'],
                 t_dim=1, u_dim=128, tu_dim=512)

def mtmt_res_emb_v0_4():
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=38), treat_feat_enc=nn.Embedding(num_embeddings=2, embedding_dim=16), task_names=['label_nextday_login'],
                 t_dim=1, u_dim=128, tu_dim=256)

def mtmt_res_emb_v0_4_0():
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=None, drop=0.2), treat_feat_enc=nn.Embedding(num_embeddings=2, embedding_dim=16), task_names=['label_nextday_login'],
                 t_dim=1, u_dim=128, tu_dim=256)

def mtmt_res_emb_v0_4_1():
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=None, drop=0.2), treat_feat_enc=nn.Embedding(num_embeddings=2, embedding_dim=16), task_names=['label_nextday_login'],
                 t_dim=16, u_dim=128, tu_dim=256)

def mtmt_res_emb_v0_5():
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=38), treat_feat_enc=nn.Identity(), task_names=['label_nextday_login'],
                 t_dim=1, u_dim=128, tu_dim=256)

def mtmt_res_emb_v0_tFeatChangeDim():
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=38), treat_feat_enc=nn.Embedding(num_embeddings=10, embedding_dim=16), task_names=['label_nextday_login'],
                 t_dim=16, u_dim=128, tu_dim=256)

def mtmt_res_emb_v0_woNorm():
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=38), treat_feat_enc=nn.Embedding(num_embeddings=10, embedding_dim=16), task_names=['label_nextday_login'],
                 t_dim=1, u_dim=128, tu_dim=256)

def mtmt_res_emb_v0_MulAttn():
    user_feat_enc_hidden_dim=16
    return MTMT(user_feat_enc=resnet18(hidden_dim=user_feat_enc_hidden_dim, out_dim=32), treat_feat_enc=MLP(in_chans=1, hidden_chans=[16, 32]), task_names=['label_nextday_login'],
                 t_dim=1, u_dim=user_feat_enc_hidden_dim * 8, tu_dim=128)

def mtmt_res_emb_v0_MulAttn0():
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=38), treat_feat_enc=nn.Embedding(num_embeddings=10, embedding_dim=38), task_names=['label_nextday_login'],
                 t_dim=1, u_dim=128, tu_dim=128)

def mtmt_res_emb_v0_transEnhance():
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=38), treat_feat_enc=nn.Embedding(num_embeddings=10, embedding_dim=16), task_names=['label_nextday_login'],
                 t_dim=1, u_dim=128, tu_dim=256)

def mtmt_res_emb_v0_normEnhance():
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=38), treat_feat_enc=nn.Embedding(num_embeddings=10, embedding_dim=16), task_names=['label_nextday_login'],
                 t_dim=1, u_dim=128, tu_dim=256, tu_enhance_norm=nn.BatchNorm1d)

def mtmt_res_emb_v1():
    user_feat_enc_hidden_dim = 64
    return MTMT(user_feat_enc=resnet18(hidden_dim=user_feat_enc_hidden_dim, out_dim=128), treat_feat_enc=nn.Embedding(num_embeddings=10, embedding_dim=16), task_names=['label_nextday_login'],
                 t_dim=1, u_dim=user_feat_enc_hidden_dim * 8, tu_dim=256)

def mtmt_res_emb_v1_0():
    user_feat_enc_hidden_dim = 32
    return MTMT(user_feat_enc=resnet18(hidden_dim=user_feat_enc_hidden_dim, out_dim=128), treat_feat_enc=nn.Embedding(num_embeddings=10, embedding_dim=16), task_names=['label_nextday_login'],
                 t_dim=1, u_dim=user_feat_enc_hidden_dim * 8, tu_dim=256)

def mtmt_res_emb_v2():
    user_feat_enc_hidden_dim = 64
    return MTMT(user_feat_enc=resnet50(hidden_dim=user_feat_enc_hidden_dim, out_dim=128), treat_feat_enc=nn.Embedding(num_embeddings=10, embedding_dim=16), task_names=['label_nextday_login'],
                 t_dim=1, u_dim=user_feat_enc_hidden_dim * 32, tu_dim=256)

def mtmt_res_emb_v2_0():
    user_feat_enc_hidden_dim = 8
    return MTMT(user_feat_enc=resnet50(hidden_dim=user_feat_enc_hidden_dim, out_dim=128), treat_feat_enc=nn.Embedding(num_embeddings=10, embedding_dim=16), task_names=['label_nextday_login'],
                 t_dim=1, u_dim=user_feat_enc_hidden_dim * 32, tu_dim=256)

def mtmt_res_mlp_v0():
    user_feat_enc_hidden_dim = 64
    return MTMT(user_feat_enc=resnet18(hidden_dim=user_feat_enc_hidden_dim, out_dim=128), treat_feat_enc=MLP(in_chans=1, hidden_chans=[16, 32, 64]), task_names=['label_nextday_login'],
                 t_dim=1, u_dim=user_feat_enc_hidden_dim * 8, tu_dim=256)

def mtmt_res_mlp_v0_0():
    user_feat_enc_hidden_dim = 16
    return MTMT(user_feat_enc=resnet18(hidden_dim=user_feat_enc_hidden_dim, out_dim=128), treat_feat_enc=MLP(in_chans=1, hidden_chans=[16]), task_names=['label_nextday_login'],
                 t_dim=1, u_dim=user_feat_enc_hidden_dim * 8, tu_dim=256)

def mtmt_res_mlp_v0_1():
    user_feat_enc_hidden_dim = 16
    return MTMT(user_feat_enc=resnet18(hidden_dim=user_feat_enc_hidden_dim, out_dim=128), treat_feat_enc=MLP(in_chans=1, hidden_chans=[16, 32]), task_names=['label_nextday_login'],
                 t_dim=1, u_dim=user_feat_enc_hidden_dim * 8, tu_dim=256)

def mtmt_cnn_emb_v0():
    return MTMT(user_feat_enc=cnn_simple(hidden_chans=[16, 32, 64, 128]), treat_feat_enc=nn.Embedding(num_embeddings=2, embedding_dim=16), task_names=['label_nextday_login'],
                 t_dim=1, u_dim=128, tu_dim=256)

def mtmt_cnn_emb_v1():
    return MTMT(user_feat_enc=cnn_bottleneck_simple(hidden_chans=[16, 32, 64, 128]), treat_feat_enc=nn.Embedding(num_embeddings=2, embedding_dim=16), task_names=['label_nextday_login'],
                 t_dim=1, u_dim=128, tu_dim=256)

def mtmt_vit_emb_v0():
    return MTMT(user_feat_enc=vit_tiny_patch2_224(img_size=622, in_chans=1), treat_feat_enc=nn.Embedding(num_embeddings=2, embedding_dim=16), task_names=['label_nextday_login'],
                 t_dim=1, u_dim=192, tu_dim=256)

def mtmt_mmoe_emb_v0(task_names=['label_nextday_login', 'label_login_days_diff'], num_cls=[1, 1]):
    #[('label_nextday_login', 1), ('label_after7_login_days', 8), ('label_login_days_diff', 13)]
    return MTMT(user_feat_enc=MMOE(encoder_class=resnet18, num_experts=4, task_names=task_names, in_feat=629, 
                            enc_kwargs={'all': {'hidden_dim': 16, 'out_dim': None}},
                            rep_grad=False), 
                 treat_feat_enc=nn.Embedding(num_embeddings=2, embedding_dim=16), task_names=task_names,
                 t_dim=1, u_dim=128, tu_dim=256, num_cls=num_cls)

def mtmt_res_cnn_emb_v0():
    # seperately process discrete feature; zero-encode and pad the discrete feature
    return MTMT(user_feat_enc=nn.ModuleList([resnet18(hidden_dim=16), cnn_simple(hidden_chans=[16, 32, 64, 128], in_chans=9, strides=[1, 1, 2, 1])]), 
                treat_feat_enc=nn.Embedding(num_embeddings=2, embedding_dim=16),
                task_names=['label_nextday_login'],
                t_dim=1, u_dim=128, tu_dim=256)

def mtmt_res_cnn_emb_v1():
    # seperately process discrete feature; zero-encode but not pad the discrete feature
    return MTMT(user_feat_enc=nn.ModuleList([resnet18(hidden_dim=16), cnn_simple(hidden_chans=[16, 32, 64, 128], in_chans=1, strides=[1, 1, 2, 1])]), 
                treat_feat_enc=nn.Embedding(num_embeddings=2, embedding_dim=16),
                task_names=['label_nextday_login'],
                t_dim=1, u_dim=128, tu_dim=256)

def mtmt_res_disc__emb_v0():
    # seperately encode discrete features and add back as the input to user_feat_enc
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, disc_encoder=DiscEncoder(embed_dim=8, out_shape='1d')),
                treat_feat_enc=nn.Embedding(num_embeddings=2, embedding_dim=16),
                task_names=['label_nextday_login'],
                t_dim=1, u_dim=128, tu_dim=256)

def mtmt_res_disc_mlp__emb_v0():
    return  MTMT(user_feat_enc=nn.ModuleList([resnet18(hidden_dim=8), 
                    DiscEncoder(embed_dim=8, out_shape='2d',
                                enc=MLP(hidden_chans=[16, 32, 64], in_chans=9, transpose=True))
                ]), 
                treat_feat_enc=nn.Embedding(num_embeddings=2, embedding_dim=16),
                task_names=['label_nextday_login'],
                t_dim=1, u_dim=64, tu_dim=128)


def mtmt_res_disc_mlp__emb_cnn_v0():
    return  MTMT(user_feat_enc=nn.ModuleList([resnet18(hidden_dim=8), 
                    DiscEncoder(embed_dim=8, out_shape='2d',
                                enc=MLP(hidden_chans=[16, 32, 64], in_chans=9, transpose=True))
                ]), 
                treat_feat_enc=cnn_simple(in_chans=1, hidden_chans=[4, 8, 16], strides=[1, 1, 1], encoder=nn.Embedding(num_embeddings=2, embedding_dim=8)),
                task_names=['label_nextday_login'],
                t_dim=16, u_dim=64, tu_dim=128)

def mtmt_res_disc_cnn__emb_v0():
    return  MTMT(user_feat_enc=nn.ModuleList([resnet18(hidden_dim=4), 
                    DiscEncoder(embed_dim=8, out_shape='2d',
                                enc=cnn_simple(hidden_chans=[12, 16, 32], in_chans=9, strides=[1, 2, 1]))
                ]), 
                treat_feat_enc=nn.Embedding(num_embeddings=2, embedding_dim=16),
                task_names=['label_nextday_login'],
                t_dim=1, u_dim=32, tu_dim=64)

def mtmt_res_disc_cnn__emb_v0_0():
    return  MTMT(user_feat_enc=nn.ModuleList([resnet18(hidden_dim=8), 
                    DiscEncoder(embed_dim=8, out_shape='2d',
                                enc=cnn_simple(hidden_chans=[16, 32, 64], in_chans=9, strides=[1, 2, 1]))
                ]), 
                treat_feat_enc=nn.Embedding(num_embeddings=2, embedding_dim=16),
                task_names=['label_nextday_login'],
                t_dim=1, u_dim=64, tu_dim=128)

def mtmt_res_disc_cnn__emb_v1():
    return MTMT(user_feat_enc=nn.ModuleList([resnet18(hidden_dim=16), 
                    DiscEncoder(embed_dim=8, out_shape='1d',
                                enc=cnn_simple(hidden_chans=[16, 32, 64, 128], in_chans=1, strides=[1, 1, 2, 1]))
                ]), 
                treat_feat_enc=nn.Embedding(num_embeddings=2, embedding_dim=16),
                task_names=['label_nextday_login'],
                t_dim=1, u_dim=128, tu_dim=256)

def mtmt_res_cnn_MLP_v0():
    # seperately process discrete feature; zero-encode and pad the discrete feature
    return MTMT(user_feat_enc=nn.ModuleList([resnet18(hidden_dim=16), MLP(in_chans=9, hidden_chans=[16, 32, 64, 128], transpose=True)]), 
                treat_feat_enc=nn.Embedding(num_embeddings=2, embedding_dim=16),
                task_names=['label_nextday_login'],
                t_dim=1, u_dim=128, tu_dim=256)

def mtmt_res__emb_MLP_v0_4_0():
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=None, drop=0.2), 
                treat_feat_enc=MLP(in_chans=1, transpose=True, hidden_chans=[4, 8, 16], drop_rate=0.1, encoder=nn.Embedding(num_embeddings=2, embedding_dim=8)),
                task_names=['label_nextday_login'], t_dim=16, u_dim=128, tu_dim=256)

def mtmt_res__emb_cnn_v0_4_0():
    return MTMT(user_feat_enc=resnet18(hidden_dim=16, out_dim=None, drop=0.2), 
                treat_feat_enc=cnn_simple(in_chans=1, hidden_chans=[4, 8, 16], strides=[1, 1, 1], encoder=nn.Embedding(num_embeddings=2, embedding_dim=8)),
                task_names=['label_nextday_login'], t_dim=16, u_dim=128, tu_dim=256)