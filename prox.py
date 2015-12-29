import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import pylab
import pdb

def div( two_layered_pix ):
    '''
    The negative conjugate of the discrede
    gradient below
    '''
    p1 = two_layered_pix[0 , : , : ]
    p2 = two_layered_pix[1 , : , : ]

    return p1 -  np.roll( p1 , -1, axis = 0 ) + p2 - np.roll( p2, -1 , axis = 1 )

def grad( pix ):
    '''
    Discrete gradient operator with peiodic
    boundary.
    '''
    
    p1 =  np.roll( pix, 1, axis = 0 )  - pix  
    p2 =  np.roll( pix, 1, axis = 1 )  - pix  
    
    p1 = p1.reshape( ( (1,) + p1.shape ) )
    p2 = p2.reshape( ( (1,) + p2.shape ) )

    return np.concatenate(  (p1, p2) )

def prox_TV( theta, sig, tau = 0.25, lam = 0.1, update = False):
    '''
    Implementation of Chambolle's algorithm
    titled An Algorithm for Total Variation Minimization
    and Applications.
    '''

    N = theta.shape[0]
    p = np.zeros( (2,) + theta.shape )
    
    for i in range(150):

        g_over_lambda = theta / lam

        term = tau * grad( div( p ) - g_over_lambda )
        
        q = ( p + term ) / ( 1 + np.abs( term ) )
        
        if update:

            # Update lambda for next round
            lam = ( N * sig ) / np.linalg.norm( div( q ) )
   
        p = q

    return theta - lam * div( p )
     
def grad_g( theta, data_hat, K_hat, K_bar_hat, sig ):
    ''' 
    Gradient (not discrete!) of the following function

          1
    ------------- || K * theta - data ||^2
    2 * sig * sig

    where * denotes convolution.
    '''

    theta_hat = np.fft.rfft2( theta )
    return 1./(sig*sig) * np.fft.irfft2( K_bar_hat * (K_hat*theta_hat - data_hat ) )
    
def fwdbckwd( theta, data_hat, sig, K_hat, K_bar_hat, alpha ): 
    '''
    Implementing the forward backward 
    algorithm with the above gradient
    and proximity mapping.
    '''
    
    x = theta
    for i in range(40):
        print i
        y = x - sig*sig* grad_g( x, data_hat, K_hat, K_bar_hat, sig )
        #x = x + 0.5*(prox_TV( y, sig, lam = alpha) - x )
        x = prox_TV( y, sig, lam = alpha*sig*sig)
 
    return x
        
def do_it( obs, K_hat, K_bar_hat, sig ):
    '''
    applying the entire majorisation-minimization
    framework from oliveira et. al.
    '''
    
    theta = obs
    data_hat = np.fft.rfft2( obs )
    N = obs.shape[0]
    
    
    for i in range(25):
        
        print
        print "Start F-B cycle number " + str(i+1)
        alpha = ( N*N + 1 )/ ( 1 + np.linalg.norm( grad( theta ) ) )

        theta = fwdbckwd( theta, data_hat, sig, K_hat, K_bar_hat, alpha )

    return theta

def prox_blur( theta, data_hat, sig, K_hat ):
    '''
    Proximity mapping of the above blur operator.
    Don't use it here but it is good to keep.
    '''
    theta_hat = np.fft.rfft2( theta )

    u_hat = (  (data_hat*K_hat)/(sig*sig)  +  theta_hat   )/(   (K_hat*K_hat)/(sig*sig) + 1   )

    return np.fft.irfft2( u_hat )
