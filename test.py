import numpy as np
import prox as px
from scipy import misc
import matplotlib.pyplot as plt
import pylab
import pdb
 
def test_conjugacy():
    '''
    Check that div == -grad^{*}
    '''

    n = 100

    # Test with integers
    u = np.arange( n*n ).reshape( (n,n) )
    p = np.arange( n*n*2 ).reshape( (2,n,n) )

    grad_u = px.grad( u )
    div_p  = -px.div( p )  
    
    prod1 = np.einsum( "ijk, ijk", grad_u, p )
    prod2 = np.einsum( "ij, ij"  , u,  div_p )

    assert  prod1  ==  prod2  

    # Test with normals
    
    u = np.random.normal( size=n*n ).reshape( (n,n) )
    p = np.random.normal( size=n*n*2 ).reshape( (2,n,n) )

    grad_u = px.grad( u )
    div_p  = -px.div( p )  
    
    prod1 = np.einsum( "ijk, ijk", grad_u, p )
    prod2 = np.einsum( "ij, ij"  , u,  div_p )

    assert abs( prod1 - prod2 ) < 1E-10

def test_cham( pic, noisy, sig ):
    
    clean = px.prox_TV( noisy, sig, tau = 0.25, update = True )

    image = pylab.figure()

    image.add_subplot( 2, 2, 1)
    pylab.imshow( pic , cmap=plt.cm.gray )
    plt.title( "Original" )
    
    image.add_subplot( 2, 2, 2)
    pylab.imshow( noisy, cmap=plt.cm.gray )
    plt.title( "Noisy" )

    image.add_subplot( 2, 2, 3)
    pylab.imshow( clean, cmap=plt.cm.gray )
    plt.title( "Denoised" )
    
    plt.savefig("test_cham.png")
    plt.close()


def test_prox_blur( pic, blurry, K_hat ):

    blurry_hat = np.fft.rfft2( blurry )
    rec = px.prox_blur( blurry, blurry_hat, 0.0001, K_hat )
    
    image = pylab.figure()
    
    image.add_subplot( 2, 2, 1)
    pylab.imshow( pic , cmap=plt.cm.gray )
    plt.title( "Original" )
    
    image.add_subplot( 2, 2, 2)
    pylab.imshow( blurry, cmap=plt.cm.gray )
    plt.title( "Blurred" )

    image.add_subplot( 2, 2, 3)
    pylab.imshow( rec, cmap=plt.cm.gray )
    plt.title( "Deblurred" )

    plt.savefig("test_blur.png")
    plt.close()
    
def test_fwdbckwd( pic, data, sig, K_hat, K_bar_hat ):
     
    data_hat = np.fft.rfft2( data )
    N        = pic.shape[0]    
    
    # Here we cheat - take the TRUE alpha
    alpha = ( N*N + 1 )/ ( 1 + np.linalg.norm( px.grad( pic ) ) )
    rec   = px.fwdbckwd( data, data_hat, sig, K_hat, K_bar_hat, alpha )
        
    
    image = pylab.figure()
    
    image.add_subplot( 2, 2, 1)
    pylab.imshow( pic , cmap=plt.cm.gray )
    plt.title( "Original" )
    
    image.add_subplot( 2, 2, 2)
    pylab.imshow( data, cmap=plt.cm.gray )
    plt.title( r"Blurred = $\theta^{(0)}$" )

    image.add_subplot( 2, 2, 3)
    pylab.imshow( rec, cmap=plt.cm.gray )
    plt.title( r"$\theta^{(\infty)}$" )

    plt.savefig("test_fwdbckwd.png")
    plt.close()
    
    
 
def test_all( pic, data, sig, K_hat, K_bar_hat ):
     
    data_hat = np.fft.rfft2( data )
    N        = pic.shape[0]    
   
    rec   = px.do_it( data, K_hat, K_bar_hat, sig )
        
    
    image = pylab.figure()
    
    image.add_subplot( 2, 2, 1)
    pylab.imshow( pic , cmap=plt.cm.gray )
    plt.title( "Original" )
    
    image.add_subplot( 2, 2, 2)
    pylab.imshow( data, cmap=plt.cm.gray )
    plt.title( r"Blurred = $\theta^{(0)}$" )

    image.add_subplot( 2, 2, 3)
    pylab.imshow( rec, cmap=plt.cm.gray )
    plt.title( r"$\theta^{(\infty)}$" )

    plt.savefig("test_all.png")
    plt.close()
    
def test_aya( data, sig, K_hat, K_bar_hat ):


    aya_r = data[:,:,0]
    aya_g = data[:,:,1]
    aya_b = data[:,:,2]

    # Denoise every component individually
    v_r = px.do_it( aya_r,  K_hat, K_bar_hat, sig )
    v_g = px.do_it( aya_g,  K_hat, K_bar_hat, sig )
    v_b = px.do_it( aya_b,  K_hat, K_bar_hat, sig )

    rec = np.empty( data.shape )
    rec[:,:,0] = v_r
    rec[:,:,1] = v_g
    rec[:,:,2] = v_b

    # Cast to integer data type because that's what plotter needs.
    rec  = rec.astype( data.dtype )
    plt.savefig("rec.png")

    # Boring Plotting
    image = pylab.figure()
    image.add_subplot( 1, 2, 1)
    pylab.imshow( data  )
    plt.title( "Original" )
    image.add_subplot( 1, 2, 2)
    pylab.imshow( rec )
    plt.title( "Enhanced" )
    misc.imsave('aya_denoise.png' , rec )
    plt.close()

   
    

if __name__ == "__main__":

    boat       = misc.imread( 'boat.png' , flatten =True) 
    camera     = misc.imread( 'camera.png' , flatten =True) 
    sig        = 0.1 # Noise amplitude
    big_sig    = 15  # Noise for chambolle.
    N          = boat.shape[0]
    
    
    # Create and take FT of kernels
    m          = 5  
    K          = np.zeros( (N,N) )
    K[0:m,0:m] = 1.0
    K          = K / np.sum( K )
    K_hat      = np.fft.rfft2( K )
    K_bar      = np.zeros( (N,N) )
    K_bar[1-m:,1-m:] = 1.0
    K_bar[0,0]       = 1.0
    K_bar[0,1-m:]    = 1.0
    K_bar[1-m:,0]    = 1.0
    K_bar            = K_bar / np.sum( K_bar )
    K_bar_hat        = np.fft.rfft2( K_bar ) 
    
    # Noisy image for Chambolle's algorithm
    big_noise = np.random.normal( loc=0, scale=big_sig, size=len(np.ravel(camera)) ).reshape( camera.shape )
    noisy_camera = camera + big_noise
        
    # Blurred noisy image for the real tests
    noise   = np.random.normal( loc = 0.0, scale = sig, size = len(np.ravel(boat)) ).reshape( boat.shape ) 
    noisy   = boat + noise # add noise
    noisy_hat = np.fft.rfft2( noisy ) # FT noisy
    boat_data = np.fft.irfft2( K_hat * noisy_hat ) # convolve with blur kernel and do iFT 

    # Tests!!!
    test_conjugacy()
    test_cham( camera, noisy_camera, big_sig )
    test_all( boat, boat_data, sig, K_hat, K_bar_hat )
    test_fwdbckwd( boat, boat_data, sig, K_hat, K_bar_hat )

    
    ################ AYA #############################
    # aya = misc.imread( 'aya_original.png' )
    # N   = aya.shape[1]
    # sig = .01
    
    # # Create and take FT of kernels
    # m          = 3  
    # K          = np.zeros( (N,N) )
    # K[0:m,0:m] = 1.0
    # K          = K / np.sum( K )
    # K_hat      = np.fft.rfft2( K )
    # K_bar      = np.zeros( (N,N) )
    # K_bar[1-m:,1-m:] = 1.0
    # K_bar[0,0]       = 1.0
    # K_bar[0,1-m:]    = 1.0
    # K_bar[1-m:,0]    = 1.0
    # K_bar            = K_bar / np.sum( K_bar )
    # K_bar_hat        = np.fft.rfft2( K_bar ) 
    
    # # Real Deal
    # test_aya( aya, sig, K_hat, K_bar_hat )

    

     
        
       
