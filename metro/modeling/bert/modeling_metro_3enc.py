"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import code
import torch
from torch import nn
from .modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPooler
from .modeling_bert import BertLayerNorm as LayerNormClass
import metro.modeling.data.config as cfg
import torch.nn.functional as F
from metro.modeling.utils import GradientReversal


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        m = nn.Sigmoid()
        x = m(x)
        return x


class METRO_Encoder(BertPreTrainedModel):

    def __init__(self, config, args):
        super(METRO_Encoder, self).__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config, args)
        self.args = args
        self.pooler = BertPooler(config)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.img_dim = config.img_feature_dim 
        '''
        self.joint_align = args.joint_align
        if joint_align:
            self.joint_query = nn.Parameter(torch.empty(1, 1, d_model))
        '''
        try:
            self.use_img_layernorm = config.use_img_layernorm
        except:
            self.use_img_layernorm = None

        self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.use_img_layernorm:
            self.LayerNorm = LayerNormClass(config.hidden_size, eps=config.img_layer_norm_eps)

        self.apply(self.init_weights)

#不懂
    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, img_feats, joint_query, vertex_query, is_train, input_ids=None, token_type_ids=None, attention_mask=None,
            position_ids=None, head_mask=None):

        batch_size = len(img_feats)
        seq_length = len(img_feats[0])
        input_ids = torch.zeros([batch_size, seq_length],dtype=torch.long).cuda()

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
#這是啥
        if attention_mask is None: 
            attention_mask = torch.ones_like(input_ids)
#這是啥
        if token_type_ids is None: 
            token_type_ids = torch.zeros_like(input_ids)

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) #增加維度
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError
#不懂
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        # Project input token features to have spcified hidden size
        img_embedding_output = self.img_embedding(img_feats)
        if is_train == True and self.args.da_mode == 'uda':
            
            if self.args.joint_align:
                joint_query = self.img_embedding(joint_query)#joint query也要降維
            if self.args.vertex_align:
                vertex_query = self.img_embedding(vertex_query)#vertex query也要降維
        #print('metro encoder joint 降維', joint_query)
        #print('metro encoder vertex 降維', vertex_query)

        # We empirically observe that adding an additional learnable position embedding leads to more stable training
        embeddings = position_embeddings + img_embedding_output #加上positional embedding

        if self.use_img_layernorm:
            embeddings = self.LayerNorm(embeddings) #layer normalization
        embeddings = self.dropout(embeddings)
        
        '''
        #所以也要從METRO_Body_Network輸入is_train變成self.training
        #tgt.shape[0]要改成joint數量
        joint_query = None
        if self.training and self.joint_align:
            joint_query = self.joint_query.expand(tgt.shape[0], -1, -1)
        '''

        #-----------------------------
        #呼叫BertEncoder
        '''
        input要加入joint_query, output也要
        '''
        encoder_outputs, joint_query, vertex_query = self.encoder(embeddings, joint_query, vertex_query, is_train,
                extended_attention_mask, head_mask=head_mask) #attention
        '''
        if self.training and self.instance_align:
            da_output['instance_query'] = instance_query
        '''
        #-----------------------------

        
        sequence_output = encoder_outputs[0]
#1st residual, 2nd layer normalization, MLP呢
#2nd residual 在METRO
        outputs = (sequence_output,)
        if self.config.output_hidden_states:
            all_hidden_states = encoder_outputs[1]
            outputs = outputs + (all_hidden_states,)
        if self.config.output_attentions:
            all_attentions = encoder_outputs[-1]
            outputs = outputs + (all_attentions,)
        '''
        最後也要return da_output'''

        return outputs, joint_query, vertex_query

class METRO(BertPreTrainedModel): #有三層METRO 也就是三層encoder block
    '''
    The archtecture of a transformer encoder block we used in METRO
    '''
    def __init__(self, config, args):
        super(METRO, self).__init__(config, args)
        self.config = config
        self.args = args
        self.bert = METRO_Encoder(config, args)
        
        self.cls_head = nn.Linear(config.hidden_size, self.config.output_feature_dim)
        self.residual = nn.Linear(config.img_feature_dim, self.config.output_feature_dim)
#甚麼是self.apply
        self.apply(self.init_weights)

        

    def forward(self, input, input_ids=None, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
            next_sentence_label=None, position_ids=None, head_mask=None):
        '''
        # self.bert has three outputs
        # predictions[0]: output tokens
        # predictions[1]: all_hidden_states, if enable "self.config.output_hidden_states"
        # predictions[2]: attentions, if enable "self.config.output_attentions"
        '''
        #print(input)
        img_feats = input['features']
        joint_query = input['joint_query']
        vertex_query = input['vertex_query']
        is_train = input['is_train']
        #---------------------------
        #呼叫METRO_Encoder
        #print('傳進METRO Encoder的input img_feats', img_feats)
        predictions, joint_query, vertex_query = self.bert(img_feats=img_feats, joint_query=joint_query, vertex_query=vertex_query, is_train = is_train, input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        #---------------------------
        #print('metro forward finish')

        # We use "self.cls_head" to perform dimensionality reduction. We don't use it for classification.
        pred_score = self.cls_head(predictions[0]) #將prediction降維到output feature dim
        #print('metro joint ', joint_query)
        #print('metro vertex ', vertex_query)
        #print('output_feature_dim ', self.config.output_feature_dim)
        if is_train == True and self.args.da_mode == 'uda':
            if self.args.joint_align and self.config.output_feature_dim != 3:
            #if self.args.joint_align:
                joint_query = self.cls_head(joint_query) #joint query也要降維
            if self.args.vertex_align and self.config.output_feature_dim != 3:
            #if self.args.vertex_align:  
                vertex_query = self.cls_head(vertex_query) #vertex query也要降維
        #print('降維metro joint ', joint_query)
        #print('降維metro vertex ', vertex_query)
        #print("-----------------------------------------------------")

        res_img_feats = self.residual(img_feats) #將input降維到output feature dim

        pred_score = pred_score + res_img_feats #residual connection

        output = {'features': pred_score, 'joint_query': joint_query, 'vertex_query': vertex_query, 'is_train': is_train}

        #print('METRO pred', output['features'])
        #print('---------')
        #print('features: ', output['features'].size())
        #print('joint_query: ', output['joint_query'].size())
        #print('vertex_query: ', output['vertex_query'].size())
        if self.config.output_attentions and self.config.output_hidden_states:
            return output, predictions[1], predictions[-1]
        else:
            return output

class METRO_Hand_Network(torch.nn.Module):
    '''
    End-to-end METRO network for hand pose and mesh reconstruction from a single image.
    '''
    def __init__(self, args, config, backbone, trans_encoder):
        super(METRO_Hand_Network, self).__init__()
        self.config = config
        self.backbone = backbone
        self.trans_encoder = trans_encoder
        self.upsampling = torch.nn.Linear(195, 778)
        self.cam_param_fc = torch.nn.Linear(3, 1)
        self.cam_param_fc2 = torch.nn.Linear(195+21, 150) 
        self.cam_param_fc3 = torch.nn.Linear(150, 3)

    def forward(self, images, mesh_model, mesh_sampler, meta_masks=None, is_train=False):
        batch_size = images.size(0)
        # Generate T-pose template mesh
        template_pose = torch.zeros((1,48))
        template_pose = template_pose.cuda()
        template_betas = torch.zeros((1,10)).cuda()
        template_vertices, template_3d_joints = mesh_model.layer(template_pose, template_betas)
        template_vertices = template_vertices/1000.0
        template_3d_joints = template_3d_joints/1000.0

        template_vertices_sub = mesh_sampler.downsample(template_vertices)

        # normalize
        template_root = template_3d_joints[:,cfg.J_NAME.index('Wrist'),:]
        template_3d_joints = template_3d_joints - template_root[:, None, :]
        template_vertices = template_vertices - template_root[:, None, :]
        template_vertices_sub = template_vertices_sub - template_root[:, None, :]
        num_joints = template_3d_joints.shape[1]

        # concatinate template joints and template vertices, and then duplicate to batch size
        ref_vertices = torch.cat([template_3d_joints, template_vertices_sub],dim=1)
        ref_vertices = ref_vertices.expand(batch_size, -1, -1)

        # extract global image feature using a CNN backbone
        image_feat = self.backbone(images)

        # concatinate image feat and template mesh
        image_feat = image_feat.view(batch_size, 1, 2048).expand(-1, ref_vertices.shape[-2], -1)
        features = torch.cat([ref_vertices, image_feat], dim=2)

        if is_train==True:
            # apply mask vertex/joint modeling
            # meta_masks is a tensor of all the masks, randomly generated in dataloader
            # we pre-define a [MASK] token, which is a floating-value vector with 0.01s
            constant_tensor = torch.ones_like(features).cuda()*0.01
            features = features*meta_masks + constant_tensor*(1-meta_masks)     

        # forward pass
        if self.config.output_attentions==True:
            features, hidden_states, att = self.trans_encoder(features)
        else:
            features = self.trans_encoder(features)

        pred_3d_joints = features[:,:num_joints,:]
        pred_vertices_sub = features[:,num_joints:,:]

        # learn camera parameters
        x = self.cam_param_fc(features)
        x = x.transpose(1,2)
        x = self.cam_param_fc2(x)
        x = self.cam_param_fc3(x)
        cam_param = x.transpose(1,2)
        cam_param = cam_param.squeeze()

        temp_transpose = pred_vertices_sub.transpose(1,2)
        pred_vertices = self.upsampling(temp_transpose)
        pred_vertices = pred_vertices.transpose(1,2)

        if self.config.output_attentions==True:
            return cam_param, pred_3d_joints, pred_vertices_sub, pred_vertices, hidden_states, att
        else:
            return cam_param, pred_3d_joints, pred_vertices_sub, pred_vertices

class METRO_Body_Network(torch.nn.Module):
    '''
    End-to-end METRO network for human pose and mesh reconstruction from a single image.
    '''
    def __init__(self, args, config, backbone, trans_encoder, mesh_sampler):
        super(METRO_Body_Network, self).__init__()
        self.config = config
        self.config.device = args.device
        self.backbone = backbone
        self.trans_encoder = trans_encoder
        self.upsampling = torch.nn.Linear(431, 1723)
        self.upsampling2 = torch.nn.Linear(1723, 6890)
        self.conv_learn_tokens = torch.nn.Conv1d(49,431+14,1)
        self.cam_param_fc = torch.nn.Linear(3, 1)
        self.cam_param_fc2 = torch.nn.Linear(431, 250)
        self.cam_param_fc3 = torch.nn.Linear(250, 3)
        self.da_mode = args.da_mode
        
        self.hidden_size = config.hidden_size
        self.joint_align = args.joint_align
        self.vertex_align = args.vertex_align
        self.backbone_align=args.backbone_align

        if self.joint_align:
            self.joint_D = MLP(self.hidden_size, self.hidden_size, 1, 3) #input size=3, hidden=32, output size=1, 3層
            for layer in self.joint_D.layers:
                nn.init.xavier_uniform_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)
            # joint_query
            self.joint_query = nn.Parameter(torch.rand(1, 1, 2051))

        if self.vertex_align:
            self.vertex_D = MLP(self.hidden_size, self.hidden_size, 1, 3) 
            for layer in self.vertex_D.layers:
                nn.init.xavier_uniform_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)
            # vertex_query
            self.vertex_query = nn.Parameter(torch.rand(1, 1, 2051))
        
        if self.backbone_align:
            self.grl = GradientReversal()
            self.backbone_D = MLP(2048, 2048, 1, 3)
            for layer in self.backbone_D.layers:
                nn.init.xavier_uniform_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)    
        

    def forward(self, images, smpl, mesh_sampler, meta_masks=None, is_train=False):
        batch_size = images.size(0)
        #print('batch size ', batch_size)
        # Generate T-pose template mesh
        template_pose = torch.zeros((1,72)) #[[0, 0, 0,..., 0]]
##不懂
        template_pose[:,0] = 3.1416 # Rectify "upside down" reference mesh in global coord   
        template_pose = template_pose.cuda(self.config.device)
        template_betas = torch.zeros((1,10)).cuda(self.config.device)
        template_vertices = smpl(template_pose, template_betas)

        # template mesh simplification
        template_vertices_sub = mesh_sampler.downsample(template_vertices)
        template_vertices_sub2 = mesh_sampler.downsample(template_vertices_sub, n1=1, n2=2)

        # template mesh-to-joint regression 
        template_3d_joints = smpl.get_h36m_joints(template_vertices)
        template_pelvis = template_3d_joints[:,cfg.H36M_J17_NAME.index('Pelvis'),:]
        template_3d_joints = template_3d_joints[:,cfg.H36M_J17_TO_J14,:]
        num_joints = template_3d_joints.shape[1]

        # normalize
        template_3d_joints = template_3d_joints - template_pelvis[:, None, :]
        template_vertices_sub2 = template_vertices_sub2 - template_pelvis[:, None, :]
        # concatinate template joints and template vertices, and then duplicate to batch size
        ref_vertices = torch.cat([template_3d_joints, template_vertices_sub2],dim=1) #將template joints跟template vertices concat起來
        ref_vertices = ref_vertices.expand(batch_size, -1, -1)
        
        # extract image feature maps using a CNN backbone
        #print('image', images)
        #print('..............................')
        image_feat = self.backbone(images)
        #print('image_feat', image_feat)
        image_feat_newview = image_feat.view(batch_size,2048,-1)
        image_feat_newview = image_feat_newview.transpose(1,2)
        # and apply a conv layer to learn image token for each 3d joint/vertex position
        img_tokens = self.conv_learn_tokens(image_feat_newview)

        # concatinate image feat and template mesh
        features = torch.cat([ref_vertices, img_tokens], dim=2) 
        
        joint_query = None
        vertex_query = None

        if is_train==True:
            # apply mask vertex/joint modeling
            # meta_masks is a tensor of all the masks, randomly generated in dataloader
            # we pre-define a [MASK] token, which is a floating-value vector with 0.01s
#不懂
            constant_tensor = torch.ones_like(features).cuda(self.config.device)*0.01 
            features = features*meta_masks + constant_tensor*(1-meta_masks)            
            if self.joint_align:
                joint_query = self.joint_query.expand(batch_size, -1, -1)
            if self.vertex_align:
                vertex_query = self.vertex_query.expand(batch_size, -1, -1)
        
        #nn.Sequential只能傳入一個參數!!!
        input = {'features': features, 'joint_query': joint_query, 'vertex_query': vertex_query, 'is_train': is_train}

        # forward pass
        if self.config.output_attentions==True:
            output, hidden_states, att = self.trans_encoder(input)
        else:

            #一般self.config.output_attentions=False
            #-------------------------
            #METRO是一個encoder block的架構 在modeling_metro.py的class METRO當中
            #print('metro body forward start')
            #print('#-------------------------')
            #print('features: ', features)
            #print('joint: ', joint_query)
            #print('vertex: ', vertex_query)
            #print('傳進METRO的input', input)
            #print('input joint query', input['joint_query'])
            #print('input vertex query', input['vertex_query'])
            output = self.trans_encoder(input)
            #-------------------------
            #img_feats, joint_query, vertex_query, input_ids=None, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
            #next_sentence_label=None, position_ids=None, head_mask=None)
        features = output['features']
        joint_query = output['joint_query']
        vertex_query = output['vertex_query']
        
        pred_3d_joints = features[:,:num_joints,:]
        pred_vertices_sub2 = features[:,num_joints:,:]
        

        if is_train and self.da_mode == 'uda':
            joint_domain_pred = None
            vertex_domain_pred = None
            backbone_domain_pred=None

            pred_3d_joints = pred_3d_joints[:batch_size//2, :, :] #source的pred joint
            pred_vertices_sub2 = pred_vertices_sub2[:batch_size//2, :, :] #source的pred vertex
            #print('output joint query ', joint_query)
            #print('output vertex query ', vertex_query)
            if self.joint_align:
                joint_domain_pred = self.joint_D(joint_query)
            if self.vertex_align:
                vertex_domain_pred = self.vertex_D(vertex_query)
            if self.backbone_align:
                backbone_domain_pred=self.backbone_D(image_feat_newview)

        # learn camera parameters
        x = self.cam_param_fc(pred_vertices_sub2)
        x = x.transpose(1,2)
        x = self.cam_param_fc2(x)
        x = self.cam_param_fc3(x)
        cam_param = x.transpose(1,2)
        cam_param = cam_param.squeeze()

        temp_transpose = pred_vertices_sub2.transpose(1,2)
        pred_vertices_sub = self.upsampling(temp_transpose)
        pred_vertices_full = self.upsampling2(pred_vertices_sub)
        pred_vertices_sub = pred_vertices_sub.transpose(1,2)
        pred_vertices_full = pred_vertices_full.transpose(1,2)

        if self.config.output_attentions==True:
            return cam_param, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices_full, hidden_states, att, joint_domain_pred, vertex_domain_pred, backbone_domain_pred
        else:
            if is_train and self.da_mode == 'uda':
            #print('body metwork pred_3d_joints ', pred_3d_joints)
                return cam_param, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices_full, joint_domain_pred, vertex_domain_pred, backbone_domain_pred
            else:
                return cam_param, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices_full