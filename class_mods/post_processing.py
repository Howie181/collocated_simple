import matplotlib.pyplot as plt
import numpy as np

class post_proc:
    def contour_plot(self, x_cor, y_cor, phi_plot):
        plt.pcolormesh(x_cor,y_cor,phi_plot,cmap='hot') #,vmin=0, vmax=1)
        plt.colorbar()
        plt.show()
    def line_plot_diag(self, x_cor, y_cor, phi_plot, nx):
        plot_diag = np.fliplr(np.transpose(phi_plot)).diagonal()
        plt.plot(x_cor[0, :], plot_diag, 'm-', marker='x', label='n=%s' % nx)
        plt.legend(loc=1, prop={'size': 14})
        plt.ylabel('$\phi$', size=14)
        plt.xlabel('$x$', size=14)
        plt.xlim(0, 1)
        plt.ylim(-5, 105)
        plt.show()
    def streamlines(self, x_cor, y_cor, u, v, mag_u, Re):
        plt.streamplot(x_cor, y_cor, u, v, density = 1.75, color=mag_u, cmap='hot', linewidth=1, broken_streamlines=False)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title('Streamline Plot, Re=%i' % Re)
        #plt.colorbar()
        plt.show()