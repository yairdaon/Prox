import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import pylab
 
def div( two_layered_pix ):
    p1 = two_layered_pix[0 , : , : ]
    p2 = two_layered_pix[1 , : , : ]

    return p1 -  np.roll( p1 , -1, axis = 0 ) + p2 - np.roll( p2, -1 , axis = 1 )


def grad( pix ):
    
    p1 =  np.roll( pix, 1, axis = 0 )  - pix  
    p2 =  np.roll( pix, 1, axis = 1 )  - pix  
    
    p1 = p1.reshape( ( (1,) + p1.shape ) )
    p2 = p2.reshape( ( (1,) + p2.shape ) )

    return np.concatenate(  (p1, p2) )

def fast_prox( data, sig, eps = 0.051, tau = 0.25):
    
    N = data.shape[0]
    lam = .1
    u   = 0 
    p = np.zeros( (2,) + data.shape )
    i = 0
    while True:

        i = i + 1
        g_over_lambda = data / lam

        term = tau * grad( div( p ) - g_over_lambda )
        
        q = ( p + term ) / ( 1 + np.abs( term ) )

        if np.linalg.norm( p - q ) < eps or i > 10000:
            break
        
        # Update lambda for next round
        lam = (N * sig ) / np.linalg.norm( div( q ) )
   
        p = q
    
    return data - lam * div( p )

if __name__ == "__main__":
    ############### Cameraman #########################
    pic     = misc.imread( 'camera.png' )
    
    # Noise amplitude
    sig = 20

    # Create the noise vector
    noise   = np.random.normal( loc = 0.0, scale = sig, size = len(np.ravel(pic)) ).reshape( pic.shape )

    # Corrupt the image with noise
    data    = pic + noise
    N       = data.shape[0]
    eps     = 0.005

    clean = fast_denoise( data, sig, eps = eps)

    image = pylab.figure()

    image.add_subplot( 2, 2, 1)
    pylab.imshow( pic , cmap=plt.cm.gray )
    plt.title( "Original" )
    
    image.add_subplot( 2, 2, 2)
    pylab.imshow( data, cmap=plt.cm.gray )
    plt.title( "Corrupted" )

    image.add_subplot( 2, 2, 3)
    pylab.imshow( clean, cmap=plt.cm.gray )
    plt.title( "Reconstructed" )


    ################ AYA #############################
    pic     = misc.imread( 'aya_original.png' )
    
    sig = 3
    eps = 0.0005
    
    pic_r = pic[:,:,0]
    pic_g = pic[:,:,1]
    pic_b = pic[:,:,2]

    # Denoise every component individually
    v_r = fast_denoise( pic_r, sig, eps=eps )
    v_g = fast_denoise( pic_g, sig, eps=eps )
    v_b = fast_denoise( pic_b, sig, eps=eps )

    rec = np.empty( pic.shape )
    rec[:,:,0] = pic_r - v_r
    rec[:,:,1] = pic_g - v_g
    rec[:,:,2] = pic_b - v_b

    # Cast to integer data type because that's what plotter needs.
    rec  = rec.astype( pic.dtype )

    # Boring Plotting
    image = pylab.figure()
    image.add_subplot( 1, 2, 1)
    pylab.imshow( pic  )
    plt.title( "Original" )
    image.add_subplot( 1, 2, 2)
    pylab.imshow( rec )
    plt.title( "Denoised" )
    misc.imsave('aya_denoise.png' , rec )














# image.add_subplot( 2, 2, 4)
# pylab.imshow( data - clean , cmap=plt.cm.gray )
# plt.title( "Noise - Correction term" )
# plt.savefig('camera_plots.png' )
# plt.close()



# def prox( data, lam, tau = 0.25, eps = 0.01 ): 
    
#     p = np.zeros( (2,) + data.shape )

#     g_over_lambda = data / lam
#     i = 0
#     while True:
#         i = i + 1
#         term = tau * grad( div( p ) - g_over_lambda )
        
#         q = ( p + term ) / ( 1 + np.abs( term ) )

#         # Consider stopping
#         if np.linalg.norm( np.ravel(p-q) ) < eps or i > 5:
#             break
        
#         # Update
#         p = q
        
#     return q

# def denoise( data, sig, eps = 0.1 ):
    
#     N = data.shape[0]
#     lam = .1
#     u   = 0 
#     i = 0
#     while True:

#         i = i + 1
#         v   = lam * div( prox( data, lam, eps = eps ) )
#         f   = np.linalg.norm( v )

#         # Update lambda for next round
#         lam = (N * sig * lam) / f
   
#         if np.linalg.norm( u - v ) < eps or i > 2:
#             break
    
#         u = v
    
#     return v

# def crop( pix ):
#     dim = min( pix.shape )

#     return pix[ 0 : dim , 0 : dim ]

