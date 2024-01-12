from misc.epc import *
import spherical as S360
import supervision as sp

def xyz2uv(xyz, eps = 0.001, tx=0.0,ty=0.0,tz=0.0):
    xyz += eps

    x, y, z = torch.unbind(xyz, dim=2)

    x = x+tx
    y = y+ty
    z = z+tz
   
    pi = float(np.pi)##torch.acos(torch.zeros(1)).item() * 2
    
    u = torch.atan2(x, -y)
    v = - torch.atan(z / torch.sqrt(x**2 + y**2)) ###  (default: - for z neg (under horizon) - grid sample instead expects -1,-1 top-left
        
   
    u = (u / pi)
    v = (2.0 * v) / pi 

    u = torch.clamp(u, min=-1, max=1)
    v = torch.clamp(v, min=-1, max=1)
        
    ###output: [batch_size x num_points x 2]##range -1,+1

    output = torch.stack([u, v], dim=-1) 
                        
    return output


def feature_splat(x_in, x_depth_in, device, max_d = 16.0, t = (0.26,0.0,0.0)): 
    ###FIXME to batch
    x_in = x_in.unsqueeze(0)#### Bx3xhxw
    x_depth_in = x_depth_in.unsqueeze(0)#### Bxhxw - invalid vlaues masked
         

    x_depth_in = x_depth_in.unsqueeze(1)#### Bx1xhxw
               
    x_shape = x_in.shape

    width  = x_shape[3]
    height = x_shape[2]

    sgrid = S360.grid.create_spherical_grid(width).to(device)
    uvgrid = S360.grid.create_image_grid(width, height).to(device) 

    tx, ty, tz = t
            
    disp = torch.cat(
                (
                    S360.derivatives.dphi_translation(sgrid, x_depth_in, tx, ty, tz),##S360.derivatives.dphi_horizontal(sgrid, x_depth_in, tx),
                    S360.derivatives.dtheta_translation(sgrid, x_depth_in, tx, ty, tz)##S360.derivatives.dtheta_horizontal(sgrid, x_depth_in, tx) ####CHECK IT - 0 values mask
                ),
                dim=1
            )
           
    render_coords = uvgrid + disp

    render_coords[:, 0, :, :] = torch.fmod(render_coords[:, 0, :, :] + width, width)
    render_coords[torch.isnan(render_coords)] = 0.0
    render_coords[torch.isinf(render_coords)] = 0.0   

    splat_rgb,splat_mask = sp.splatting.render(x_in.to(device), x_depth_in.to(device), render_coords, max_depth=max_d)           
        
    return splat_rgb, splat_mask


def transform_map(img_in, depth_in,  t = (0,0,0)):
    print('in shape',img_in.shape, depth_in.shape)
        
    epc = EPC(gpu=False,YZ_swap = False)#### 

    depth_in = depth_in.unsqueeze(0)

    ###epc input: B,H,W
    pw = epc(depth_in)
    ###epc output: B,3,H,W

    S = pw.shape

    pw = pw.reshape(S[0], S[1],-1).permute(0,2,1) ### Bx(h*w)x2

    tx, ty, tz = t
            
    uv_inputs = xyz2uv(pw, tx=tx,ty=ty,tz=tz)
        
    uv_inputs = uv_inputs.unsqueeze(1) ###Bx1xh*w,2    
    
    b_img_in = img_in.unsqueeze(0) #### Bx3xhxw

    b_img_in = torch.roll(b_img_in, S[3]//4, 3) ####FIXMEEEE adjust for F.grid coords
                
    output = F.grid_sample(b_img_in, uv_inputs, align_corners=True).squeeze(2)### Bx3xh*w

    ish = b_img_in.shape

    output = output.reshape(ish) ##### Bx3xhxw

    ##output = torch.roll(output, ish[3]//4, 3)

    ##print('grid',output.shape)

    x_out = output

    return x_out

def test_target_translation(root_dir, ith): 
    device = torch.device('cpu') 

    dataset = PNVS_Dataset(root_dir=root_dir, return_name=True, full_size_rgb=True)
    if args.ith >= 0:
        to_visualize = [dataset[ith]]
    else:
        to_visualize = dataset

    for x_img, x_depth, x_camera, ti0, ti1, ti2, tc0, tc1, tc2, f_name in tqdm(to_visualize):
       
        f_name = os.path.split(f_name)[-1].split('.')[0]
   
        print('loading image', f_name)

        print('x shape', x_img.shape)

        print('src camera',x_camera)
                                            
                
        x_img_c = x2image(x_img.to(device) )  
                                       
        src_depth = x_depth.unsqueeze(0).unsqueeze(0).to(device)

        plt.figure(args.ith)
        plt.title(f_name+' full GT')
        plt.imshow(x_img_c)          
                    
         
        #plt.figure(args.ith+1007)
        #plt.title(f_name+' depth')
        #plt.imshow(src_depth.squeeze(0).squeeze(0)) 
        

        test_targets = True

        if(test_targets):
            print('test shape t0',ti0.shape,tc0.shape)
            ti0_c = x2image(ti0.to(device) )
            print('ti0 cam',tc0)
            plt.figure(args.ith+1011)
            plt.title(f_name+' t0')
            plt.imshow(ti0_c) 

            ti1_c = x2image(ti1.to(device) )
            print('ti1 cam',tc1)
            plt.figure(args.ith+1012)
            plt.title(f_name+' t1')
            plt.imshow(ti1_c)

            ti2_c = x2image(ti2.to(device) )
            print('ti2 cam',tc2)
            plt.figure(args.ith+1013)
            plt.title(f_name+' t2')
            plt.imshow(ti2_c)


        t0_delta = (tc0 - x_camera) / 1000.0
        t1_delta = (tc1 - x_camera) / 1000.0
        t2_delta = (tc2 - x_camera) / 1000.0

        
        print('to delta meters',t0_delta, t1_delta, t2_delta)
        
        test_translation = True
        
        if(test_translation):        
                        

            img_tr, occ_tr = feature_splat(x_img, x_depth, device, t = (t2_delta[:,0],t2_delta[:,1],t2_delta[:,2]))

            ####FIXMEEEEE
            ###img_tr = transform_map(x_img, x_depth,t = (t1_delta[:,0],t1_delta[:,1],t1_delta[:,2])) ####Bx3xhxw

            img_c_tr_masked = ( (occ_tr*img_tr).squeeze(0).numpy().transpose([1, 2, 0])*255).astype(np.uint8) 
            img_c_tr        = ( (img_tr).squeeze(0).numpy().transpose([1, 2, 0])*255).astype(np.uint8)

            occ_tr_c = occ_tr.squeeze(0).squeeze(0).numpy().astype(np.uint8)    
            
            
                    
              
            plt.figure(args.ith+1005)
            plt.title(f_name+' full GT translated masked')
            plt.imshow(img_c_tr_masked)  

            plt.figure(args.ith+1006)
            plt.title(f_name+' full GT translated')
            plt.imshow(img_c_tr)
               
        
            plt.figure(args.ith+1008)
            plt.title(f_name+' mask GT translated')
            plt.imshow(occ_tr_c)  

        
                
        plt.show()

def test_features_translation(root_dir, ith): 
    device = torch.device('cpu') 

    feats_translation = FeatsTranslation()

    dataset = PNVS_Dataset(root_dir=root_dir, return_name=True, full_size_rgb=True, target_id = 1)
    if args.ith >= 0:
        to_visualize = [dataset[ith]]
    else:
        to_visualize = dataset

    for x_img, x_depth, x_camera, ti, tc, f_name in tqdm(to_visualize):
       
        f_name = os.path.split(f_name)[-1].split('.')[0]
   
        print('loading image', f_name)

        print('x shape', x_img.shape)

        print('src camera',x_camera)
                                            
                
        x_img_c = x2image(x_img.to(device) )  
                                       
        src_depth = x_depth.unsqueeze(0).unsqueeze(0).to(device)

        plt.figure(args.ith)
        plt.title(f_name+' full GT')
        plt.imshow(x_img_c)          
                    
         
        #plt.figure(args.ith+1007)
        #plt.title(f_name+' depth')
        #plt.imshow(src_depth.squeeze(0).squeeze(0)) 
        

        test_targets = True

        if(test_targets):
            print('test shape t0',ti.shape, tc.shape)
            ti_c = x2image(ti.to(device) )
            print('ti cam',tc)
            plt.figure(args.ith+1011)
            plt.title(f_name+' ti')
            plt.imshow(ti_c) 
                        

        t_delta = (tc - x_camera) / 1000.0
        
        
        print('to delta meters',t_delta)

        print('t_delta',t_delta.shape)
        
        test_translation = True
        
        if(test_translation):       
            
            x_in = x_img.unsqueeze(0)#### Bx3xhxw
            x_depth_in = x_depth.unsqueeze(0)#### Bxhxw - invalid vlaues masked
         

            x_depth_in = x_depth_in.unsqueeze(1)#### Bx1xhxw


            b, c, h, w = x_in.size()     
                        
                
            sgrid = S360.grid.create_spherical_grid(w).to(x_depth_in.device)
            uvgrid = S360.grid.create_image_grid(w, h).to(x_depth_in.device) 

            tr = (t_delta[:,0],t_delta[:,1],t_delta[:,2])

            img_tr, occ_tr = feats_translation(x_in, x_depth_in, sgrid, uvgrid, t_delta)

            ####FIXMEEEEE
            ###img_tr = transform_map(x_img, x_depth,t = (t1_delta[:,0],t1_delta[:,1],t1_delta[:,2])) ####Bx3xhxw

            img_c_tr_masked = ( (occ_tr*img_tr).squeeze(0).numpy().transpose([1, 2, 0])*255).astype(np.uint8) 
            img_c_tr        = ( (img_tr).squeeze(0).numpy().transpose([1, 2, 0])*255).astype(np.uint8)

            occ_tr_c = occ_tr.squeeze(0).squeeze(0).numpy().astype(np.uint8)                        
                    
              
            #plt.figure(args.ith+1005)
            #plt.title(f_name+' full GT translated masked')
            #plt.imshow(img_c_tr_masked)  

            plt.figure(args.ith+1006)
            plt.title(f_name+' full GT translated')
            plt.imshow(img_c_tr)
               
        
            #plt.figure(args.ith+1008)
            #plt.title(f_name+' mask GT translated')
            #plt.imshow(occ_tr_c)  

        
                
        plt.show()