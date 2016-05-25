# -*- coding: utf-8 -*-
"""
Created on Fri May 20 02:23:14 2016

@author: Polka
"""
from __future__ import division

import scipy
import scipy.io 
import scipy.signal
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import thread
import pywt
import math
import imresize
from collections import defaultdict


def tree(): return defaultdict(tree)
###############################################################################
#ICA functions

def make_patches(data,patch_size):
    
    m,n = np.shape(data)  
    num_to_divide = (m / patch_size[0], n / patch_size[1])
    
    horizontal_slices = np.split(data, num_to_divide[0], axis=0)
    
    array_of_patches = []
    
    for sl in horizontal_slices:
        array_of_patches.append((np.split(sl, num_to_divide[1], axis=1))) 
    return array_of_patches
    
def image_from_patch_matrix(rec_patches, patch_size):
    
    patches_per_side = int(np.sqrt(len(rec_patches)))
    vertical_slice = []
    for v in range(patches_per_side):
        horizontal_slice = []
        for h in range(patches_per_side):
            horizontal_slice.append(rec_patches[v*patches_per_side+h].reshape((patch_size)))
        vertical_slice.append(np.concatenate(horizontal_slice, axis=1))
    reconstructed_image = np.concatenate(vertical_slice, axis=0)
    return reconstructed_image

def G_str(y):
    return np.tanh(y)
        
def G_double_str(y):
    return 1 - (np.tanh(y))**2        

def ICA_basis(input_image):
    #ICs=230
    """
    Independent Component Analysis of an image.
    algorithm of a book 'Image processing: Fundamentals', p.274
    """
    patch_size = (16,16)
    
    #input_mean = np.mean(input_image)
    #data = input_image - input_mean
    
    patches = make_patches(input_image, patch_size)
    I = np.shape(patches)[0]*np.shape(patches)[1] #whole number of patches
            
    #Step 1: Form a matrix P of patches 
    vertical_patches = []
    for h_slice in patches:
        for patch in h_slice:
            vertical_patches.append(patch.reshape(patch_size[0]*patch_size[1],1))
            
    #Step 0: Remove the mean of each patch        
    P = np.array(np.concatenate((vertical_patches), axis = 1))
    patch_mean = [np.mean(P[:,i]) for i in range(np.shape(P)[1])]
    P = P - patch_mean
    
    #Step 2: Compute the average M of P and remove it from each vector to make
    # P_tilda matrix    
    mean = np.mean(P) #here mean is zero, but this step exists in the book
    P_tilda = P - mean
    
    #Step 3: Compute autocorrelation matrix C
    C = (1 / I) * np.dot(P_tilda, P_tilda.T)
    
    #Step 4: Compute the nonzero eigenvalues of C in decreasing order  
    eig_vals, eig_vecs = np.linalg.eig(C)
    eig_vals = np.real(eig_vals)
    eig_vecs = np.real(eig_vecs)    
    
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = [eig_pairs[i] for i in range(len(eig_vals)) if (np.abs(eig_pairs[i][0]) > 0.0002)] #remove zero eigs    
    eig_pairs.sort(key=lambda tup: tup[0], reverse=True) #sorts by decreasing
    E = len(eig_pairs) #get a number of nonzero eigenvalues
    eig_vecs = [eig_pairs[i][1] for i in range(len(eig_pairs))]
    normalised_eig_vecs = [eig_pairs[i][1]/(np.sqrt(eig_pairs[i][0])) \
                           for i in range(len(eig_pairs))]
                          
    #Step 5: Make normalized matrix U_tilda of eigenvectors
    U_tilda = np.array(normalised_eig_vecs)
    
    #Step 6: make matrix Q_tilda
    Q_tilda = np.dot(U_tilda, P_tilda)
    
    #Launching a cycle to work out all vectors of W matrix
    np.random.seed(4)
    w_vectors = []
    for e in range(E):
        #print "----------Started %s E---------------------------------------" % e
        #Step 7:Select randomly an Ex1 vector
        w = np.random.uniform(-1,1,E).reshape(E,1)
        
        #Step 8: Normalise vector w_1 to unit form
        w_tilda = w / np.sqrt(np.sum(w*w))
        
        converge = False
        while not converge:
            #Step 9: Project all data vectors q_tilda on w_tilda
            Y = np.dot(w_tilda.T, Q_tilda)
            
            #Step 10: Update components of w_tilda
            w_plus = w_tilda * (1 / I) * np.sum(G_double_str(Y)) - \
                     (1 / I) * np.sum(np.dot(Q_tilda, G_str(Y.T)))
            
            #Step 10.5: extra step to take orthogonal w's
            if e > 0:
                B = np.array(np.concatenate((w_vectors), axis = 1))
                w_plus = w_plus - np.dot(np.dot(B, B.T), w_plus)            
            
            #Step 11: Normalise vector w_plus
            w_plus_tilda = w_plus / np.sqrt(np.sum(w_plus*w_plus))
            
            #Step 12: Check whether w_plus_tilda and w_1_tilda are close enough
            if np.abs(np.dot(w_plus_tilda.T, w_tilda)) > 0.9999:
                #two vectors are identical and w_plus_tilda is the axis of ICA
                w_vectors.append(w_plus_tilda)
                converge = True
            else:
                #print np.abs(np.dot(w_plus_tilda.T, w_tilda))
                #print " repeat cycle for %s" % e
                w_tilda = w_plus_tilda
            
    W = np.array(np.concatenate((w_vectors), axis = 1))
    
    #Step 13: Project P_tilda on unscaled eigenvector matrix U
    U = np.array(eig_vecs)
    Q = np.dot(U, P_tilda)
    
    #Step 14: Compute the matrix Z
    Z = np.dot(W.T, Q)
    
    #Step 14.5: Construct vectors of matrix V_tilda
    V_tilda = np.dot(U.T, W)
    
    
    #Step 14.6: Reconstruction of ith patch
#    rec_patches = []
#    for patch in range(I):
#        rec_patch = np.dot(V_tilda, Z[:,patch])
#        rec_patch += patch_mean[patch]
#        rec_patches.append(rec_patch)
#    
#    reconstructed_image = image_from_patch_matrix(rec_patches, patch_size)
    
    basis_components = {'I':I,
                        'E':E,
                        'patch_size':patch_size,
                        'Z':Z,
                        'V_tilda':V_tilda,
                        'patch_mean':patch_mean,
                        'input_image':input_image}
       
    return basis_components

  
def MSE(C, S):
    res = np.zeros(3, dtype='float32')
    C = C.astype(np.int32)
    S = S.astype(np.int32)
    diff = C - S
    res = np.sum(diff*diff) / np.size(C)
    return res
    
def basis(V):
    plt.figure()
    
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow(V[:,i].reshape((4,4)),interpolation='nearest', cmap = 'gray')
        plt.xlabel('%s' % i)
        plt.axis('off')
    plt.show()
    return
    
def component_ranger(basis): 
    importance = []
    for remove in range(basis['E']):
        shortened_V_tilda = np.delete(basis['V_tilda'],remove,axis=1)
        shortened_Z = np.delete(basis['Z'],remove,axis=0)
        rec_patches = []
        for patch in range(basis['I']):
            rec_patch = np.dot(shortened_V_tilda, shortened_Z[:,patch])
            rec_patch += basis['patch_mean'][patch]
            rec_patches.append(rec_patch)

        reconstructed_image = image_from_patch_matrix(rec_patches, basis['patch_size'])
        importance.append(MSE(basis['input_image'],reconstructed_image))
    return importance

def remove_n_ic(basis, importance, remove): 
    #sort components by importance
    V_imp_tuple = sorted(zip(importance, basis['V_tilda'].T))
    sorted_V = np.array([V_imp_tuple[i][1] for i in range(len(importance))]).T    
    Z_imp_tuple = sorted(zip(importance, basis['Z']))
    sorted_Z = np.array([Z_imp_tuple[i][1] for i in range(len(importance))])
    
    #removing components
    shortened_V = sorted_V[:,remove:]
    shortened_Z = sorted_Z[remove:,:]
    
    rec_patches = []
    for patch in range(basis['I']):
        rec_patch = np.dot(shortened_V, shortened_Z[:,patch])
        rec_patch += basis['patch_mean'][patch]
        rec_patches.append(rec_patch)

    reconstructed_image = image_from_patch_matrix(rec_patches, basis['patch_size'])

    return reconstructed_image
    
def ICA_preparing(steganogram):
    preparations = []
    for ch in range(3):
        chan_basis = ICA_basis(steganogram[...,ch])
        chan_importance = component_ranger(chan_basis)
        preparations.append((chan_basis, chan_importance))
    return preparations
    
def ICA_filtering(preparations, remove):
    
    filtered = np.zeros((512,512,3), 'uint8')
    
    for channel in range(3):
        basis = preparations[channel][0]
        importance = preparations[channel][1]
        filt_channel = remove_n_ic(basis, importance, remove)
        filtered[...,channel] = filt_channel.clip(min=0)
                 
    return filtered    
###############################################################################

def PCA(data, PCs):

    #m,n = np.shape(data)
    n = 512
    mean = np.sum(data)
    mean /= 262144


    X = data - mean
    Z = 1/np.sqrt(n-1)*X.T
    covZ = np.dot(Z.T,Z)
    
    U,S,V = np.linalg.svd(covZ)
    
    VV = V[:PCs]
    Y = np.dot(VV,X)
    XX = np.dot(VV.T, Y)
    XX += mean

    return XX
    
def metrics(C, S):

    '''
    Function for computing the metric for comparison initial and processed stegodata
    after destruction (short variant)
    
    C      - initial stegodata
    S      - processed stegodata after destruction
    %Res 	- array with values of image metrics
    %		  Res -> [MSE IF NCC CD PSNR SSIM]
    %
    %Metrics for comparison the images
    %
    %MSE    - mean square error
    %IF     - image fidelity
    %NCC    - Normalized cross-correlation
    %CD     - Czenakowski distance (Warning! Scalar for grauscale and true color images)
    %PSNR   - peak signal-to-noise ratio
    %SSIM   - structure similarity
    '''
    
    # Control of inputted data
    #
    ShapeC = np.shape(C)
    #ShapeS = np.shape(S)
    
    ## Compute the image metrics
    #
    
    #Definition of color plane amount 
    #if (ismatrix(C))
     #   K = 1;
    #else
    K = 3

    
    #Predefine auxiliary variables
    eps = 1.1921e-07
    MSE   = np.zeros(3, dtype='float32')
    NCC   = np.zeros(3, dtype='float32')
    PSNR  = np.zeros(3, dtype='float32')
    IF    = np.zeros(3, dtype='float32')
    SSIM  = np.zeros(3, dtype='float32')
    Temp1 = np.zeros([ShapeC[0], ShapeC[1]], dtype='float32') + eps/2
    Temp2 = np.zeros([ShapeC[0], ShapeC[1]], dtype='float32') + eps
    
    #Change the type of inputted data for correct image computing
    C = C.astype(np.int32)
    S = S.astype(np.int32)
    
    # Compensation of possible negative values (shift range from [MIN;MAX] to [1;MAX-MIN+1])
    if ((np.amin(C) < 0.0) or (np.amin(S) < 0.0)):
        C = C - np.amin(C);
        S = S - np.amin(S);
    #Computation the image metrics
    for i in range(K):
        Ctemp = C[:,:,i] #Change the type of inputted data for correct image computing
        Stemp = S[:,:,i] #Change the type of inputted data for correct image computing
        CtempEnergy = np.sum(Ctemp**2)
        
        C_S_diff = Ctemp - Stemp
        	
        IF[i]   = 1 - ((np.sum(C_S_diff**2)) / CtempEnergy)
        MSE[i]  = np.sum(C_S_diff*C_S_diff) / np.size(Ctemp)
        #NCC[i]  = np.sum(Ctemp*Stemp) / CtempEnergy
        try:
            PSNR[i] = 10*np.log10(255**2/MSE[i])
        except ZeroDivisionError:
            PSNR[i] = float('inf')
        	
        #Auxiliary computation for CD metric
        Temp1 = Temp1 + np.minimum(Ctemp,Stemp, dtype='float32')
        Temp2 = Temp2 + (Ctemp+Stemp)
        	
        #Auxiliary computation for SSIM metric
        MeanC = np.mean(Ctemp)
        MeanS = np.mean(Stemp)
        VarC  = np.std(Ctemp, dtype='float32')
        VarS  = np.std(Stemp, dtype='float32')
        VarCS = np.multiply(Ctemp - MeanC,Stemp - MeanS).sum() / (Ctemp.size)
        if (VarC == 0) or (VarS == 0):
            NCC[i] = 0
        else:
            NCC[i] =  VarCS / (VarC*VarS)     
        
        SSIM[i] = (NCC[i]) * ((2*MeanC*MeanS) / (MeanC**2 + MeanS**2)) * ((2*VarC*VarS) / (VarC**2 + VarS**2))
        
        if (SSIM[i] > 1):
            SSIM[i] = 1;
        elif (SSIM[i] < 0):
            SSIM[i] = np.absolute(SSIM[i]);
            
        del Ctemp,Stemp,C_S_diff
    
    CD = np.float32(np.sum(1 - (2*Temp1)/Temp2) / (np.size(C) / K))
    
    # Forming output variable
    #
    metrics = {'MSE':MSE,
               'IF':IF,
               'NCC':NCC,
               'CD':CD,
               'PSNR':PSNR,
               'SSIM':SSIM}
    return metrics
    
def filtering(steganogram, function, parameter):
    
    filtered = np.zeros((512,512,3), 'uint8')
    stdoutmutex = thread.allocate_lock()
    exitmutexes = [False] * 3
    
    def filter_channel(function, parameter, channel):
        if function=='median':
            filt_channel = scipy.ndimage.filters.median_filter(steganogram[:,:,channel], parameter, mode='reflect')
        elif function == 'wiener':
            filt_channel = scipy.signal.wiener(steganogram[:,:,channel],parameter)
        elif function == 'gauss':
            filt_channel = scipy.ndimage.gaussian_filter(steganogram[:,:,channel],parameter, mode='reflect')
        elif function =='PCA':
            filt_channel = PCA(steganogram[:,:,channel],parameter)

        stdoutmutex.acquire()
        filtered[...,channel] = filt_channel.clip(min=0)
        stdoutmutex.release()
        exitmutexes[channel] = True # сигнал главному потоку
        return filt_channel
        
    for ch in range(3):
        thread.start_new_thread(filter_channel, (function, parameter, ch))
    while False in exitmutexes: pass
                 
    return filtered
 
###############################################################################   
def routine(container, stegodata, steganogram, extracted_raw, aux, p, g, \
            func, filt_param, method, preparations):
    if func == 'ICA':
        filtered = ICA_filtering(preparations, filt_param)                
    else:
        filtered = filtering(steganogram, func, filt_param)
        
    CCM = metrics(container, steganogram)
    FCM = metrics(container, filtered)
    
    # Now extract filtered stegodata
    if method=='dey':
        stegodata = scipy.misc.imresize(stegodata, aux[0])
        extracted_filt_dey = \
        Dey_Extraction(container, filtered, aux, p, g) 
        
        CSM = metrics(stegodata, extracted_raw)
        FSM = metrics(stegodata, extracted_filt_dey)
        
    elif method=='agarwal':
        stegodata = scipy.misc.imresize(stegodata, aux['Stego_Size'])
        extracted_filt_ag = \
        Agarwal_Extraction(container, filtered, aux, p, g) 

        CSM = metrics(stegodata, extracted_raw)
        FSM = metrics(stegodata, extracted_filt_ag)
        
    elif method=='joseph':
        stegodata = scipy.misc.imresize(stegodata, aux['Stego_Size'])
        extracted_filt_joseph = \
        Joseph_Extraction(container,filtered, aux, p, g)

        CSM = metrics(stegodata, extracted_raw)
        FSM = metrics(stegodata, extracted_filt_joseph)
        
    return CCM, FCM, CSM, FSM  
    
    
#def method_calc(pic_key,stegodata_key, method, container, stegodata, payload, filters, \
#                embedding_functions, extraction_functions, output_q, output_lock):
#    worker_res = {}
#    for p in payload: 
#        for g in method[1]:
#            steganogram, aux = embedding_functions[method[0]](container, stegodata, p, g)
#            extracted_raw = extraction_functions[method[0]](container, steganogram, aux, p, g)
#            for filt in filters:
#                for param in filt[1]:
#                    metr_C, metr_S = routine(container, steganogram, extracted_raw, \
#                                             aux, p, g, filt[0], param, method[0])
#                    
#                    
#                    
#                    worker_res[(pic_key, stegodata_key, p, method[0], g, filt[0],param, 'C')] = metr_C 
#                    worker_res[(pic_key, stegodata_key, p, method[0], g, filt[0],param, 'S')] = metr_S 
#                    output_lock.acquire()
#                    print 'finished for %s image, %s stego, %s payload, %s method, %s G, %s filter with %s parameter' % (pic_key, stegodata_key, p, method[0], g, filt[0],param)
#                    output_lock.release()
#    output_q.put(worker_res)
#                    #time.sleep(0.0001)
#    output_q.task_done()

def Fc(CM, FM):
    summ = 0
    metrics = ('MSE','IF', 'NCC', 'PSNR', 'SSIM')
    for metric in metrics:
        for chan in range(3):
            if FM[metric][chan] == 'inf':
                FM[metric][chan] = 50
                
            summ += (np.abs(CM[metric][chan] - FM[metric][chan]) / CM[metric][chan])
            #print ( (np.abs(CM[metric][chan] - FM[metric][chan]) / CM[metric][chan]))
    summ += (np.abs(CM['CD'] - FM['CD']) / CM['CD']) 
    #print ((np.abs(CM['CD'] - FM['CD']) / CM['CD']))
    
    return summ / 16
    
def Fs(CM, FM):
    summ = 0
    metrics = ('MSE','IF', 'NCC', 'PSNR', 'SSIM')
    for metric in metrics:
        for chan in range(3):
            if FM[metric][chan] == 'inf':
                FM[metric][chan] = 50
                
            summ += (np.abs(CM[metric][chan] - FM[metric][chan]) / max(CM[metric][chan],FM[metric][chan]))
            #print ((np.abs(CM[metric][chan] - FM[metric][chan]) / max(CM[metric][chan],FM[metric][chan])))
    summ += (np.abs(CM['CD'] - FM['CD']) / max(CM['CD'], FM['CD'])) 
    #print ((np.abs(CM[metric][chan] - FM[metric][chan]) / max(CM[metric][chan],FM[metric][chan])))
    
    return summ / 16 
    
def Dey_Embedding(Cont,St,Payload,G):
    
    """    
    Function for modelling the Dey embedding method (based on DWT, haar-wavelet)
    
     Cont    - container for data embedding (true color image);
     St      - stegodata/watermark (true color image)
     Payload - level of contained filling, e.g. 10% <-> 0.1
     G       - parameters of transparency
     STEG    - formed watermark/steganogram (structure):
               'Steganograms'     - 4D-array with formed steganograms
               'Stego_Size'       - 2D matrix with size of stegodata for each
                                    embedding level
               'Embedding_Method' - used algorithm of stegodata embedding
               'Payload'          - 1D-vector woth level of container filling, 
                                    e.g. 10% <-> 0.1
               'G'                - gain factor for stegodata
     AUX    - auxiliary variables for stegodata extraction (structure):
               'Embedding_Method' - used algorithm of stegodata embedding
               'Wavelet_Name'     - name of used wavelet during 2D-DWT
    """
    
    ### CONTROL OF INPUTTED DATA
    assert(St.ndim == 3),                    'Error. Stegodata should be represented as true color image.';
    assert(Cont.ndim == 3),                  'Error. Cover image should be represented as true color image.';
    assert(isinstance(Payload,float) and (Payload > 0)),\
    'Error. You should represent the level of container filling as float from 0 to1.0';
    assert(isinstance(G,float) and (G > 0)), 'Error. You should input the attenuation coefficient for stegodata as scalar.';
        
    
    ### PREPARATION PROCEDURES
    
    # Predefine auxiliay variables
    SizeC        = list(Cont.shape);
    wName        = 'db1';
    WaveSize     = np.divide(SizeC[0:2],2);
    Stego_Size   = np.zeros([1,2],'uint16');    
    
    # Redefiniton of stego proportion for correct scaling
    St = imresize.imresize(St,WaveSize,interp = 'bicubic');
    SizeS = list(St.shape);
    
    # forming the output array with formed stego for embedding
    StArray = np.zeros(SizeC+[1],St.dtype);
    
    ContCap = Payload * np.prod(SizeC[0:2]);
    StCap   = np.prod(SizeS[0:2]);
    
    # Estimation of required scaling coefficient for stegodata
    Eta       = math.sqrt(ContCap/StCap);
    StRes     = imresize.imresize(St,Eta,interp = 'bicubic');
    SizeStRes = list(StRes.shape);
    StResCap  = np.prod(SizeStRes[0:2]);
    
    # Step of image resizing
    UpScale   = 0.01;
    DownScale = 0.01;
    
    if ((np.absolute(ContCap - StResCap)/ContCap) <= 0.05) and (ContCap >= StResCap):
        StRes = np.pad(StRes, ((0,SizeC[0]-SizeStRes[0]),(0,SizeC[1]-SizeStRes[1]),(0,0)), mode = 'constant');
        StArray[:,:,:,0] = StRes[0:SizeC[0],0:SizeC[1],:];
        Stego_Size[0,:] = SizeStRes[0:2];
    else:
        while(True):
            StResUp   = imresize.imresize(StRes,1+UpScale,interp = 'bicubic');
            StResDown = imresize.imresize(StRes,1-DownScale,interp = 'bicubic');
            
            SizeStResUp   = list(StResUp.shape);
            SizeStResDown = list(StResDown.shape);
            
            if (SizeStResDown[0] <= 0) or (SizeStResDown[1] <= 0):
                DownScale = 0;
                
            StCapUp   = np.prod(SizeStResUp[0:2]);
            StCapDown = np.prod(SizeStResDown[0:2]);
            
            if (((ContCap - StCapUp)/ContCap) < 0.03) and (ContCap >= StCapUp):
                StResUp = np.pad(StResUp,((0,SizeC[0]-SizeStResUp[0]),(0,SizeC[1]-SizeStResUp[1]),(0,0)),mode = 'constant');
                StArray[:,:,:,0] = StResUp[0:SizeC[0],0:SizeC[1],:];
                Stego_Size[0,:] = SizeStResUp[0:2];
                break;
            elif (((ContCap - StCapDown)/ContCap) < 0.03) and (ContCap >= StCapDown):
                StResDown = np.pad(StResDown,((0,SizeC[0]-SizeStResDown[0]),(0,SizeC[1]-SizeStResDown[1]),(0,0)),mode = 'constant');
                StArray[:,:,:,0] = StResDown[0:SizeC[0],0:SizeC[1],:];
                Stego_Size[0,:] = SizeStResDown[0:2];
                break;
                
            UpScale += 0.005;
            DownScale += 0.005;
            
            del StResDown,StResUp,StCapDown,StCapUp;
    
    del StRes;
    
    # Create predefined array for results of DWT for cover image
    WaveSize = np.append(WaveSize,Cont.ndim);
    ContcA   = np.zeros(WaveSize,dtype = np.single);
    ContcH   = np.zeros(WaveSize,dtype = np.single);
    ContcV   = np.zeros(WaveSize,dtype = np.single);
    ContcD   = np.zeros(WaveSize,dtype = np.single);
    
    for i in range(Cont.ndim):
        ContcA[:,:,i],(ContcH[:,:,i],ContcV[:,:,i],ContcD[:,:,i]) = pywt.dwt2(Cont[:,:,i],wName);
    
    ### EMBEDDING PROCESS
    
    # Intialize the embedding process
    steganogram = np.zeros(SizeC,dtype = Cont.dtype); # dtype = np.single
    for i in range(Cont.ndim):
        
        cAS,(cHS,cVS,cDS) = pywt.dwt2(StArray[:,:,i,0],wName);
        
        cAR = (1-G) * ContcA[:,:,i] + G * cAS;
        cHR = (1-G) * ContcH[:,:,i] + G * cHS;
        cVR = (1-G) * ContcV[:,:,i] + G * cVS;
        cDR = (1-G) * ContcD[:,:,i] + G * cDS;
        
        steganogram[:,:,i] = pywt.idwt2((cAR,(cHR,cVR,cDR)),wName);
        del cAR,cHR,cVR,cDR,cAS,cHS,cVS,cDS;
            
    del ContcA,ContcH,ContcV,ContcD;
    
    ### FORMING OUTPUT RESULTS
    
    # RETURN THE RESULTS
    
    return steganogram, Stego_Size
    
    
# Define AUX variables    
def Dey_Extraction(Cont,Steganogram, Stego_Size, Payload, G):
    
    """    
    Function for modelling the Dey extraction method (based on DWT)
     
     Cont  - container for data embedding;
     STEG    - formed watermark/steganogram (structure):
               'Steganograms'     - 4D-array with formed steganograms
               'Stego_Size'       - 2D matrix with size of stegodata for each
                                    embedding level
               'Embedding_Method' - used algorithm of stegodata embedding
               'Payload'          - 1D-vector woth level of container filling, 
                                    e.g. 10% <-> 0.1
               'G'                - gain factor for stegodata
     AUX    - auxiliary variables for stegodata extraction (structure):
               'Embedding_Method' - used algorithm of stegodata embedding
               'Wavelet_Name'     - name of used wavelet during 2D-DWT
     Extr  - extracted stegodata (structure):
             'StegoData'        - 4D-array with extracted stegodata
             'Stego_Size'       - 2D matrix with size of stegodata for each
                                  embedding level
             'Embedding_Method' - used algorithm of stegodata embedding
             'Payload'          - 1D-vector woth level of container filling, 
                                    e.g. 10% <-> 0.1
             'G'                - gain factor for stegodata
    """
    
    ### CONTROL OF INPUTTED DATA
    assert(Cont.ndim == 3),        'Error. Stegodata should be represented as true color image.';
    
    wName = 'db1'
    
    ### PREPARATION PROCEDURES
    SizeC = list(Cont.shape);
    SizeCd2 = np.divide(SizeC[0:2],2).tolist();
    cA = np.zeros(SizeCd2+[Cont.ndim],dtype = np.single);
    cH = np.zeros(SizeCd2+[Cont.ndim],dtype = np.single);
    cV = np.zeros(SizeCd2+[Cont.ndim],dtype = np.single);
    cD = np.zeros(SizeCd2+[Cont.ndim],dtype = np.single);
    
    for i in range(Cont.ndim):
        cA[:,:,i],(cH[:,:,i],cV[:,:,i],cD[:,:,i]) = pywt.dwt2(Cont[:,:,i],wName);
        
    # Predefine output variable
    Stego_Data = np.zeros(Stego_Size[0,:].tolist() + [Cont.ndim], dtype = Cont.dtype);
    
    ### EXTRACTION PROCESS
    for i in range(Cont.ndim):
        cAW,(cHW,cVW,cDW) = pywt.dwt2(Steganogram[:,:,i],wName);
        
        cAS = (cAW-(1-G)*cA[:,:,i]) / G;
        cHS = (cHW-(1-G)*cH[:,:,i]) / G;
        cVS = (cVW-(1-G)*cV[:,:,i]) / G;
        cDS = (cDW-(1-G)*cD[:,:,i]) / G;
        
        Steg_Reconstruct = pywt.idwt2((cAS,(cHS,cVS,cDS)), wName);
        Steg_Reconstruct = Steg_Reconstruct[0:Stego_Size[0,0],0:Stego_Size[0,1]];
        # Steg_Reconstruct = np.clip(Steg_Reconstruct,0,255).astype(Cont.dtype);
        Stego_Data[:,:,i] = Steg_Reconstruct;
        
        del cAW,cHW,cVW,cDW,cAS,cHS,cVS,cDS,Steg_Reconstruct;
    del cA,cH,cV,cD;
    
    
    ### RETURN THE RESULTS
    return Stego_Data    

def Joseph_Embedding(Cont,St,Payload,G):
    
    """    
    Function for modelling the Joseph method (based on DWT + SVD, haar-wavelet)
    
     Cont    - container for data embedding (true color image);
     St      - stegodata/watermark (true color image)
     Payload - level of contained filling, e.g. 10% <-> 0.1
     G       - parameters of transparency
     STEG    - formed watermark/steganogram (structure):
               'Steganograms'     - 4D-array with formed steganograms
               'Stego_Size'       - 2D matrix with size of stegodata for each
                                    embedding level
               'Embedding_Method' - used algorithm of stegodata embedding
               'Payload'          - 1D-vector woth level of container filling, 
                                    e.g. 10% <-> 0.1
               'G'                - gain factor for stegodata
     AUX    - auxiliary variables for stegodata extraction (structure):
               'Embedding_Method' - used algorithm of stegodata embedding
               'sA'           - auxiliary array with LL subband of DWT decomposition the stego
               'sD'           - auxiliary array with HH subband of DWT decomposition the stego
               'sHU'          - auxiliary array with U-component of SVD decomposition the LH
               'sHV'          - auxiliary array with V-component of SVD decomposition the LH
               'sVU'          - auxiliary array with U-component of SVD decomposition the HL
               'sVV'          - auxiliary array with V-component of SVD decomposition the HL
               'Mask'         - auxiliary additive noise for stegodata (increasing the convergence of SVD)
               'Wavelet_Type' - used type of wavelet
    """
    
    ### CONTROL OF INPUTTED DATA
    assert(St.ndim == 3),                     'Error. Stegodata should be represented as true color image.';
    assert(Cont.ndim == 3),                   'Error. Cover image should be represented as true color image.';
    assert(isinstance(Payload,float) and (Payload > 0)),\
    'Error. You should represent the level of container filling as float from 0 to 1.0';
    assert(isinstance(G, float) and (G > 0)), 'Error. You should input the attenuation coefficient for stegodata as scalar.';
        
    
    ### PREPARATION PROCEDURES
    
    # Predefine auxiliay variables
    SizeC        = list(Cont.shape);
    SizeS        = list(St.shape);
    wName        = 'db1';
    SizeSW1      = np.divide(SizeC[0:2],2**1); # size of stego-image at 1st level of DWT
    SizeSW2      = np.divide(SizeC[0:2],2**2); # size of stego-image at 2nd level of DWT
    Stego_Size   = np.zeros([1,2],'uint16');    
    Steganogram  = np.zeros(SizeC,dtype = np.single); # dtype = Cont.dtype
    Mask         = (10*np.random.rand(SizeSW1[0],SizeSW1[1],St.ndim)).astype(St.dtype);
    
    # Redefiniton of stego proportion for correct scaling
    St = imresize.imresize(St,SizeSW1,interp = 'bicubic');
    SizeS = list(St.shape);

    # forming the output array with formed stego for embedding
    StArray = np.zeros(SizeSW1.tolist()+[St.ndim],St.dtype);
    
    ContCap = Payload * np.prod(SizeSW1[0:2]);
    StCap   = np.prod(SizeS[0:2]);
    
    # Estimation of required scaling coefficient for stegodata
    Eta       = math.sqrt(ContCap/StCap);
    StRes     = imresize.imresize(St,Eta,interp = 'bicubic');
    SizeStRes = list(StRes.shape);
    StResCap  = np.prod(SizeStRes[0:2]);
    
    # Step of image resizing
    UpScale   = 0.01;
    DownScale = 0.01;
    
    if ((np.absolute(ContCap - StResCap)/ContCap) <= 0.05) and (ContCap >= StResCap):
        StRes = np.pad(StRes, ((0,SizeSW1[0]-SizeStRes[0]),(0,SizeSW1[1]-SizeStRes[1]),(0,0)), mode = 'constant');
        StArray[:,:,:] = StRes[0:SizeSW1[0],0:SizeSW1[1],:];
        Stego_Size[0,:] = SizeStRes[0:2];
    else:
        while(True):
            StResUp   = imresize.imresize(StRes,1+UpScale,interp = 'bicubic');
            StResDown = imresize.imresize(StRes,1-DownScale,interp = 'bicubic');
            
            SizeStResUp   = list(StResUp.shape);
            SizeStResDown = list(StResDown.shape);
            
            if (SizeStResDown[0] <= 0) or (SizeStResDown[1] <= 0):
                DownScale = 0;
                
            StCapUp   = np.prod(SizeStResUp[0:2]);
            StCapDown = np.prod(SizeStResDown[0:2]);
            
            if (((ContCap - StCapUp)/ContCap) < 0.03) and (ContCap >= StCapUp):
                StResUp = np.pad(StResUp,((0,SizeSW1[0]-SizeStResUp[0]),(0,SizeSW1[1]-SizeStResUp[1]),(0,0)),mode = 'constant');
                StArray[:,:,:] = StResUp[0:SizeSW1[0],0:SizeSW1[1],:];
                Stego_Size[0,:] = SizeStResUp[0:2];
                break;
            elif (((ContCap - StCapDown)/ContCap) < 0.03) and (ContCap >= StCapDown):
                StResDown = np.pad(StResDown,((0,SizeSW1[0]-SizeStResDown[0]),(0,SizeSW1[1]-SizeStResDown[1]),(0,0)),mode = 'constant');
                StArray[:,:,:] = StResDown[0:SizeSW1[0],0:SizeSW1[1],:];
                Stego_Size[0,:] = SizeStResDown[0:2];
                break;
                
            UpScale += 0.005;
            DownScale += 0.005;
            
            del StResDown,StResUp,StCapDown,StCapUp;
    
    del StRes;
    
    # Create predefined array for results of DWT for cover image
    cH  = np.zeros(SizeSW1.tolist() + [Cont.ndim],dtype = np.single);
    cV  = np.zeros(SizeSW1.tolist() + [Cont.ndim],dtype = np.single);
    cD  = np.zeros(SizeSW1.tolist() + [Cont.ndim],dtype = np.single);
    ccA = np.zeros(SizeSW2.tolist() + [Cont.ndim],dtype = np.single);
    ccD = np.zeros(SizeSW2.tolist() + [Cont.ndim],dtype = np.single);
    cSHLu = np.zeros([SizeSW2[0],SizeSW2[0],Cont.ndim],dtype = np.single);
    cSHLs = np.zeros([SizeSW2[0],SizeSW2[1],Cont.ndim],dtype = np.single);
    cSHLv = np.zeros([SizeSW2[1],SizeSW2[1],Cont.ndim],dtype = np.single);
    cSLHu = np.zeros([SizeSW2[0],SizeSW2[0],Cont.ndim],dtype = np.single);
    cSLHs = np.zeros([SizeSW2[0],SizeSW2[1],Cont.ndim],dtype = np.single);
    cSLHv = np.zeros([SizeSW2[1],SizeSW2[1],Cont.ndim],dtype = np.single);
    
    
    for i in range(Cont.ndim):
        cA,(cH[:,:,i],cV[:,:,i],cD[:,:,i]) = pywt.dwt2(Cont[:,:,i],wName);        
        ccA[:,:,i],(ccH,ccV,ccD[:,:,i])    = pywt.dwt2(cA,wName);
        
        cSHLu[:,:,i],HLs,cSHLv[:,:,i] = np.linalg.svd(ccH,full_matrices = True);
        cSLHu[:,:,i],LHs,cSLHv[:,:,i] = np.linalg.svd(ccV,full_matrices = True);
        
        cSHLs[:,:,i] = np.pad(np.diag(HLs),((0,0),(0,SizeSW2[1]-len(HLs))),mode = 'constant');
        cSLHs[:,:,i] = np.pad(np.diag(LHs),((0,0),(0,SizeSW2[1]-len(LHs))),mode = 'constant');
        
        del cA,ccH,ccV;
        
    # Define the auxiliary arrays for StegU and StegS
    sA  = np.zeros([SizeSW2[0],SizeSW2[1],St.ndim],dtype = np.single);
    sD  = np.zeros([SizeSW2[0],SizeSW2[1],St.ndim],dtype = np.single);
    sHU = np.zeros([SizeSW2[0],SizeSW2[0],St.ndim],dtype = np.single);
    sHV = np.zeros([SizeSW2[1],SizeSW2[1],St.ndim],dtype = np.single);
    sVU = np.zeros([SizeSW2[0],SizeSW2[0],St.ndim],dtype = np.single);
    sVV = np.zeros([SizeSW2[1],SizeSW2[1],St.ndim],dtype = np.single);
    
    ### EMBEDDING PROCESS
    
    # Intialize the embedding process
    for j in range(Cont.ndim):
        
        # DWT decompositon of stego
        sA[:,:,j],(sH,sV,sD[:,:,j]) = pywt.dwt2(StArray[:,:,j] + Mask[:,:,j],wName);
        
        # SVD decomposition of stego (cH)
        sHU[:,:,j],sHS,sHV[:,:,j] = np.linalg.svd(sH,full_matrices = True);
        sVU[:,:,j],sVS,sVV[:,:,j] = np.linalg.svd(sV,full_matrices = True);            
        
        # data embedding
        sHS = np.pad(np.diag(sHS),((0,0),(0,SizeSW2[1]-len(sHS))),mode = 'constant');
        sVS = np.pad(np.diag(sVS),((0,0),(0,SizeSW2[1]-len(sVS))),mode = 'constant');
        
        Hf = cSHLs[:,:,j] + G*sHS;
        Vf = cSLHs[:,:,j] + G*sVS;
        
        # Inverse SVD
        Hff = np.dot(cSHLu[:,:,j],np.dot(Hf,cSHLv[:,:,j]));
        Vff = np.dot(cSLHu[:,:,j],np.dot(Vf,cSLHv[:,:,j]));
        
        # Inverse DWT
        Affc = pywt.idwt2((ccA[:,:,j],(Hff,Vff,ccD[:,:,j])),wName);
        
        # Reconstruct the image in spatial domain
        Steganogram[:,:,j] = pywt.idwt2((Affc,(cH[:,:,j],cV[:,:,j],cD[:,:,j])),wName);
        del Affc,Hff,Vff,Hf,Vf,sHS,sVS,sH,sV;
            
    del cH,cV,cD,ccA,ccD,cSHLu,cSHLs,cSHLv,cSLHu,cSLHs,cSLHv;
    
    ### FORMING OUTPUT RESULTS
    
    AUX  = {'Stego_Size':Stego_Size,\
            'Wavelet_Name':wName,\
            'sA':sA,\
            'sD':sD,\
            'sHU':sHU,\
            'sHV':sHV,\
            'sVU':sVU,\
            'sVV':sVV,\
            'Mask':Mask};
    
    # RETURN THE RESULTS
    
    return Steganogram, AUX;
    
def Joseph_Extraction(Cont,Steganogram,AUX, Payload, G):
    
    """    
    Function for modelling the Joseph extraction method (based on DWT + SVD)
     
     Cont  - container for data embedding;
     STEG    - formed watermark/steganogram (structure):
               'Steganograms'     - 4D-array with formed steganograms
               'Stego_Size'       - 2D matrix with size of stegodata for each
                                    embedding level
               'Embedding_Method' - used algorithm of stegodata embedding
               'Payload'          - 1D-vector woth level of container filling, 
                                    e.g. 10% <-> 0.1
               'G'                - gain factor for stegodata
     AUX    - auxiliary variables for stegodata extraction (structure):
               'Embedding_Method' - used algorithm of stegodata embedding
               'sA'           - auxiliary array with LL subband of DWT decomposition the stego
               'sD'           - auxiliary array with HH subband of DWT decomposition the stego
               'sHU'          - auxiliary array with U-component of SVD decomposition the LH
               'sHV'          - auxiliary array with V-component of SVD decomposition the LH
               'sVU'          - auxiliary array with U-component of SVD decomposition the HL
               'sVV'          - auxiliary array with V-component of SVD decomposition the HL
               'Mask'         - auxiliary additive noise for stegodata (increasing the convergence of SVD)
               'Wavelet_Type' - used type of wavelet
     Extr  - extracted stegodata (structure):
             'StegoData'        - 4D-array with extracted stegodata
             'Stego_Size'       - 2D matrix with size of stegodata for each
                                  embedding level
             'Embedding_Method' - used algorithm of stegodata embedding
             'Payload'          - 1D-vector woth level of container filling, 
                                    e.g. 10% <-> 0.1
             'G'                - gain factor for stegodata
    """
    
    ### CONTROL OF INPUTTED DATA
    assert(Cont.ndim == 3),        'Error. Stegodata should be represented as true color image.';

    
    ### PREPARATION PROCEDURES
    SizeC   = list(Cont.shape);
    SizeSW2 = np.divide(SizeC[0:2],2**2); # size of stego-image at 2nd level of DWT
    
    
    # Forming the reference array with DWT-SVD transform of Cont
    cSHLs = np.zeros([SizeSW2[0],SizeSW2[1],Cont.ndim],dtype = np.single);
    cSLHs = np.zeros([SizeSW2[0],SizeSW2[1],Cont.ndim],dtype = np.single);
    
    for i in range(Cont.ndim):
        cA,(cH,cV,cD) = pywt.dwt2(Cont[:,:,i],AUX['Wavelet_Name']);
        ccA,(ccH,ccV,ccD) = pywt.dwt2(cA,AUX['Wavelet_Name']);
        del cA,cH,cV,cD,ccA,ccD;
        
        HLU,HLs,HLV = np.linalg.svd(ccH,full_matrices = True);
        LHU,LHs,LHV = np.linalg.svd(ccV,full_matrices = True);
        cSHLs[:,:,i] = np.pad(np.diag(HLs),((0,0),(0,SizeSW2[1]-len(HLs))),mode = 'constant');
        cSLHs[:,:,i] = np.pad(np.diag(LHs),((0,0),(0,SizeSW2[1]-len(LHs))),mode = 'constant');
        del HLU,HLV,LHU,LHV,HLs,LHs,ccH,ccV;
        
    # Forming the reference array with DWT-SVD transform of WM
    wSHLs = np.zeros([SizeSW2[0],SizeSW2[1],Cont.ndim],dtype = np.single);
    wSLHs = np.zeros([SizeSW2[0],SizeSW2[1],Cont.ndim],dtype = np.single);

    for j in range(Cont.ndim):
        wA,(wH,wV,wD) = pywt.dwt2(Steganogram[:,:,j],AUX['Wavelet_Name']);
        wwA,(wwH,wwV,wwD) = pywt.dwt2(wA,AUX['Wavelet_Name']);
        del wH,wV,wD,wwA,wwD;
    
        HLU,HLs,HLV = np.linalg.svd(wwH,full_matrices = True);
        LHU,LHs,LHV = np.linalg.svd(wwV,full_matrices = True);
        wSHLs[:,:,j] = np.pad(np.diag(HLs),((0,0),(0,SizeSW2[1]-len(HLs))),mode = 'constant');
        wSLHs[:,:,j] = np.pad(np.diag(LHs),((0,0),(0,SizeSW2[1]-len(LHs))),mode = 'constant');
        del HLU,HLV,LHU,LHV,HLs,LHs;
        
    # Predefine output variable
    Stego_Data = np.zeros(AUX['Stego_Size'][0,:].tolist() + [Cont.ndim], dtype = Cont.dtype);
    
    ### EXTRACTION PROCESS
    for j in range(Cont.ndim):

        # Extraction the SVD components
        StHS = (wSHLs[:,:,j] - cSHLs[:,:,j]) / G;
        StVS = (wSLHs[:,:,j] - cSLHs[:,:,j]) / G;
        
        # Reconstruct DWT components of stego          
        sH = np.dot(AUX['sHU'][:,:,j], np.dot(StHS, AUX['sHV'][:,:,j]));
        sV = np.dot(AUX['sVU'][:,:,j], np.dot(StVS, AUX['sVV'][:,:,j]));
        
        # Reconstruct stego in spatial domain
        Steg_Reconstruct = pywt.idwt2((AUX['sA'][:,:,j],(sH,sV,AUX['sD'][:,:,j])), AUX['Wavelet_Name']);
        Steg_Reconstruct = Steg_Reconstruct.astype(Cont.dtype) - AUX['Mask'][:,:,j];
        Steg_Reconstruct = Steg_Reconstruct[0:AUX['Stego_Size'][0,0],0:AUX['Stego_Size'][0,1]];
        # Steg_Reconstruct = np.clip(Steg_Reconstruct,0,255);
        Stego_Data[:,:,j] = Steg_Reconstruct;
        
        del Steg_Reconstruct,sH,sV;
    del cSHLs,cSLHs,wSHLs,wSLHs;
    
    return Stego_Data;
    
def Agarwal_Embedding(Cont,St,Payload,G):
    
    """
    Function for modelling the Agarwal (SVD) method of data embedding in image
    
     Cont    - container for data embedding (true color image);
     St      - stegodata/watermark (true color image)
     Payload - level of contained filling, e.g. 10% <-> 0.1
     G       - parameters of transparency
     STEG    - formed watermark/steganogram (structure):
               'Steganograms'     - 4D-array with formed steganograms
               'Stego_Size'       - 2D matrix with size of stegodata for each
                                    embedding level
               'Embedding_Method' - used algorithm of stegodata embedding
               'Payload'          - 1D-vector woth level of container filling, 
                                    e.g. 10% <-> 0.1
               'G'                - gain factor for stegodata
     AUX     - auxiliary variables for stegodata extraction (structure):
               'Embedding_Method' - used algorithm of stegodata embedding
               'StegU'            - auxiliary array which consists from U-components of
                                    svd-decomposition the scaled stego-image
               'StegV'            - auxiliary array which consists from V-components of
                                    svd-decomposition the scaled stego-image
    """
    
    ### CONTROL OF INPUTTED DATA
    assert(St.ndim == 3),                     'Error. Stegodata should be represented as true color image.';
    assert(Cont.ndim == 3),                   'Error. Cover image should be represented as true color image.';
    assert(isinstance(Payload,float) and (Payload > 0)),\
    'Error. You should represent the level of container filling as float from 0 to1.0';
    assert(isinstance(G, float) and (G > 0)), 'Error. You should input the attenuation coefficient for stegodata as scalar.';
        
    
    ### PREPARATION PROCEDURES
    
    # Predefine auxiliay variables
    SizeC        = list(Cont.shape);
    SizeSscaled  = np.divide(SizeC[0:2],4);
    Steganogram = np.zeros(SizeC,dtype = np.single); # dtype = Cont.dtype
    Stego_Size   = np.zeros([1,2],'uint16');    
    
    # Redefiniton of stego proportion for correct scaling
    St    = imresize.imresize(St,SizeSscaled,interp = 'bicubic');
    SizeS = list(St.shape);    
    
    # forming the output array with formed stego for embedding
    StArray = np.zeros(SizeC,dtype = np.single);
    
    ContCap = Payload * np.prod(SizeC[0:2]);
    StCap   = np.prod(SizeS[0:2]);
    
    # Estimation of required scaling coefficient for stegodata
    Eta       = math.sqrt(ContCap/StCap);
    StRes     = imresize.imresize(St,Eta,interp = 'bicubic');
    SizeStRes = list(StRes.shape);
    StResCap  = np.prod(SizeStRes[0:2]);
    
    # Step of image resizing
    UpScale   = 0.01;
    DownScale = 0.01;
    
    if ((np.absolute(ContCap - StResCap)/ContCap) <= 0.05) and (ContCap >= StResCap):
        StRes = np.pad(StRes, ((0,SizeC[0]-SizeStRes[0]),(0,SizeC[1]-SizeStRes[1]),(0,0)), mode = 'constant');
        StArray[:,:,:] = StRes[0:SizeC[0],0:SizeC[1],:];
        Stego_Size[0,:] = SizeStRes[0:2];
    else:
        while(True):
            StResUp   = imresize.imresize(StRes,1+UpScale,interp = 'bicubic');
            StResDown = imresize.imresize(StRes,1-DownScale,interp = 'bicubic');
            
            SizeStResUp   = list(StResUp.shape);
            SizeStResDown = list(StResDown.shape);
            
            if (SizeStResDown[0] <= 0) or (SizeStResDown[1] <= 0):
                DownScale = 0;
                
            StCapUp   = np.prod(SizeStResUp[0:2]);
            StCapDown = np.prod(SizeStResDown[0:2]);
            
            if (((ContCap - StCapUp)/ContCap) < 0.03) and (ContCap >= StCapUp):
                StResUp = np.pad(StResUp,((0,SizeC[0]-SizeStResUp[0]),(0,SizeC[1]-SizeStResUp[1]),(0,0)),mode = 'constant');
                StArray[:,:,:] = StResUp[0:SizeC[0],0:SizeC[1],:];
                Stego_Size[0,:] = SizeStResUp[0:2];
                break;
            elif (((ContCap - StCapDown)/ContCap) < 0.03) and (ContCap >= StCapDown):
                StResDown = np.pad(StResDown,((0,SizeC[0]-SizeStResDown[0]),(0,SizeC[1]-SizeStResDown[1]),(0,0)),mode = 'constant');
                StArray[:,:,:] = StResDown[0:SizeC[0],0:SizeC[1],:];
                Stego_Size[0,:] = SizeStResDown[0:2];
                break;
                
            UpScale += 0.005;
            DownScale += 0.005;
            
            del StResDown,StResUp,StCapDown,StCapUp;
    
    del ContCap,StRes;
    
    # Create predefined array for results of SVD for cover image
    ContU  = np.zeros([SizeC[0],SizeC[0],SizeS[2]],dtype = np.single);
    ContSc = np.zeros([SizeC[0],SizeC[1],SizeS[2]],dtype = np.single);
    ContV  = np.zeros([SizeC[1],SizeC[1],SizeS[2]],dtype = np.single);
    
    for i in range(Cont.ndim):
        ContU[:,:,i],S,ContV[:,:,i] = np.linalg.svd(Cont[:,:,i],full_matrices = True);
        ContSc[:,:,i] = np.pad(np.diag(S),((0,0),(0,SizeC[1]-len(S))),mode = 'constant');
        del S;
        
    # Define the auxiliary arrays for StegU and StegS
    StegU = np.zeros([SizeC[0],SizeC[0],SizeS[2]],dtype = np.single);
    StegV = np.zeros([SizeC[1],SizeC[1],SizeS[2]],dtype = np.single);
    
    ### EMBEDDING PROCESS
    
    # Intialize the embedding process
    for i in range(Cont.ndim):
    
        StegU[:,:,i], S, StegV[:,:,i] = np.linalg.svd(StArray[:,:,i],full_matrices = True);
        
        Ss = ContSc[:,:,i] + (G * np.pad(np.diag(S),((0,0),(0,SizeC[1]-len(S))),mode = 'constant'));
        
        Steganogram[:,:,i] = np.dot(ContU[:,:,i],np.dot(Ss,ContV[:,:,i]));
        del S, Ss;
            
    del ContU,ContV,ContSc;
    
    ### FORMING OUTPUT RESULTS
    
    AUX  = {'Stego_Size':Stego_Size,'StegU':StegU,'StegV':StegV};
    
    # RETURN THE RESULTS
    
    return Steganogram, AUX;
    
def Agarwal_Extraction(Cont, Steganogram, AUX, Payload, G):
    
    """    
    Function for modelling the Agarwal method of data extraction from image (based on SVD)
     
     Cont  - container for data embedding;
     STEG    - formed watermark/steganogram (structure):
               'Steganograms'     - 4D-array with formed steganograms
               'Stego_Size'       - 2D matrix with size of stegodata for each
                                    embedding level
               'Embedding_Method' - used algorithm of stegodata embedding
               'Payload'          - 1D-vector woth level of container filling, 
                                    e.g. 10% <-> 0.1
               'G'                - gain factor for stegodata
     AUX     - auxiliary variables for stegodata extraction (structure):
               'Embedding_Method' - used algorithm of stegodata embedding
               'StegU'            - auxiliary array which consists from U-components of
                                    svd-decomposition the scaled stego-image
               'StegV'            - auxiliary array which consists from V-components of
                                    svd-decomposition the scaled stego-image
     Extr  - extracted stegodata (structure):
             'StegoData'        - 4D-array with extracted stegodata
             'Stego_Size'       - 2D matrix with size of stegodata for each
                                  embedding level
             'Embedding_Method' - used algorithm of stegodata embedding
             'Payload'          - 1D-vector woth level of container filling, 
                                    e.g. 10% <-> 0.1
             'G'                - gain factor for stegodata
    """
    
    ### CONTROL OF INPUTTED DATA
    assert(Cont.ndim == 3),        'Error. Stegodata should be represented as true color image.';
    assert(isinstance(AUX,dict)),  'Error. You should input AUX as dictionary.';

    
    ### PREPARATION PROCEDURES
    SizeC = list(Cont.shape);
    Stego_Data = np.zeros(AUX['Stego_Size'][0,:].tolist() + [Cont.ndim], dtype = Cont.dtype);
    
    # Forming 'bearing' svd-components of container
    ContSc = np.zeros(SizeC, dtype = np.float);
    for i in range(Cont.ndim):
        U,S,V = np.linalg.svd(Cont[:,:,i].astype(np.float));
        ContSc[:,:,i] = np.pad(np.diag(S),((0,0),(0,SizeC[1]-len(S))),mode = 'constant');
        del U,S,V;
    
    ### EXTRACTION PROCESS
    
    for i in range(Cont.ndim):

        U,SW,V = np.linalg.svd(Steganogram[:,:,i].astype(np.float),full_matrices = True);
        
        SW = np.pad(np.diag(SW),((0,0),(0,SizeC[1]-len(SW))),mode = 'constant');
        Sr = (SW - ContSc[:,:,i]) / G;
        
        Steg_Reconstruct = np.dot(AUX['StegU'][:,:,i],np.dot(Sr,AUX['StegV'][:,:,i]));
        Steg_Reconstruct = Steg_Reconstruct[0:AUX['Stego_Size'][0,0],0:AUX['Stego_Size'][0,1]];
        # Steg_Reconstruct = np.clip(Steg_Reconstruct,0,255).astype(Cont.dtype);
        Stego_Data[:,:,i] = Steg_Reconstruct;
        
        del U,V,Sr,SW,Steg_Reconstruct;
    del ContSc;
    
    
    ### RETURN THE RESULTS
    return Stego_Data;    
    