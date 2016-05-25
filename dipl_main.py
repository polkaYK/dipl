# -*- coding: utf-8 -*-
"""
Created on Wed May 25 13:37:28 2016

@author: Polka
"""
from __future__ import division

import scipy
import scipy.io 
import scipy.signal
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
import sys

import pickle

import dipl_functions as f


#Setting values
#if __name__ == '__main__':
print '256'
packet_name = '50_100_packet'
#packet_fullname = os.path.normpath('G:/flickr/packet_mat/'+packet_name)################
packet = scipy.io.loadmat(packet_name)############################################

#stego_name = os.path.normpath('G:/flickr/packet_mat/stego')##########################
stego_pack = scipy.io.loadmat('stego')###############################################
print '263'
payload = (0.05,0.1,0.15,0.2,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95)
#payload = (0.05,)
G_DA = (0.02, 0.04, 0.06, 0.08)
G_J = (0.5, 1.0, 1.5, 2.0)

windows = (3,5,7)
sigma = (0.1,0.5,1)
pcs = (32,64,96,128,160,192,224,256,288,320,352,384,416,448,480)
#pcs = (32,)
ics = (0,16,32,48,64,80,96,112,128,144,160,176,192,208,224,240)

filters = [('median', windows), ('wiener', windows), ('gauss',sigma), ('PCA', pcs), ('ICA', ics)]
methods = [('dey',G_DA), ('agarwal',G_DA), ('joseph', G_J)]
embedding_functions = {'dey': f.Dey_Embedding,
                       'agarwal': f.Agarwal_Embedding,
                       'joseph': f.Joseph_Embedding}
                       
extraction_functions = {'dey': f.Dey_Extraction,
                       'agarwal': f.Agarwal_Extraction,
                       'joseph': f.Joseph_Extraction}

# launch a cycle of pictures, then payloads, then G(because there are two G's), STEGOS
results = f.tree()
num_pics = 1 ##FOR example
res_folder = 'results/'
print '289'
sys.stdout.flush()
for i in range(num_pics): #Cycle of pictures
    pic_key = 'im60'   ###+str(i)
    container = packet[pic_key]
    
    for j in range(1):  #Cycle of stegoimages 
        stegodata_key = 'stego3'   # +str(j+1)
        stegodata = stego_pack[stegodata_key]
        
        for method in methods: #Cycle of stegomethods
            
            for g in method[1]: #Cycle of G's
                
                for p in payload: #Cycle of payloads
                    steganogram, aux = \
                            embedding_functions[method[0]](container, stegodata, p, g)
                    extracted = \
                            extraction_functions[method[0]](container, steganogram, aux, p, g) 
                    
                    for filt in filters:
                        if filt[0] == 'ICA':
                            preparations = f.ICA_preparing(steganogram)        
                        else:
                            preparations = None
                            
                        for param in filt[1]:
                            # metrics
                            CCM, FCM, CSM, FSM = f.routine(container, stegodata,\
                                                           steganogram, extracted,\
                                                           aux, p, g, filt[0], param,\
                                                           method[0], preparations)
                            # Calculating functionals                               
                            FC = f.Fc(CCM, FCM) 
                            FS = f.Fs(CSM, FSM)
                            
                            metric_dict = {'CCM':CCM,
                                           'FCM':FCM,
                                           'CSM':CSM,
                                           'FSM':FSM,
                                           'FC' :FC,
                                           'FS' :FS}
                            #saving results
                            results[pic_key][stegodata_key][method[0]][g][p][filt[0]][param] = metric_dict
                            
                        pickle.dump(results, open( res_folder+pic_key+stegodata_key+method[0]+str(g)+str(p)+filt[0]+'.p', "wb" ))
                        print '--------------SAVED '+pic_key+stegodata_key+method[0]+str(g)+str(p)+filt[0]
                        
                    pickle.dump(results, open( res_folder+pic_key+stegodata_key+method[0]+str(g)+str(p)+'.p', "wb" )) 
                    print '--------------SAVED '+pic_key+stegodata_key+method[0]+str(g)+str(p)
                    
                pickle.dump(results, open( res_folder+pic_key+stegodata_key+method[0]+str(g)+'.p', "wb" ))
                print '--------------SAVED '+pic_key+stegodata_key+method[0]+str(g)
                
            pickle.dump(results, open( res_folder+pic_key+stegodata_key+method[0]+'.p', "wb" ))     
            print '--------------SAVED '+pic_key+stegodata_key+method[0]
               
        pickle.dump(results, open( res_folder+pic_key+stegodata_key+'.p', "wb" ))  
        print '--------------SAVED '+pic_key+stegodata_key
                              
    pickle.dump(results, open( res_folder+pic_key+'.p', "wb" ))                        
    print '--------------SAVED '+pic_key
    
print '!!!!!!!!!!!!!!!!!!!!!!FINISHED SUCCESSFULLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
        

