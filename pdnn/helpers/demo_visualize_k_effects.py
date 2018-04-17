from artemis.general.mymath import cosine_distance

from pdnn.helpers.pid_encoder_decoder import lowpass_random, pid_encode, pid_decode, Herder
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec


def demo_visualize_k_effects(
        a_list=np.arange(0.0, 1.0, 0.001),
        rows=2,  # Rows for grid plotting
        cols=6,  # Columns for grid plotting
        cutoff=0.005,
        n_samples=550,
        s_as_triangles=False,
        seed=1234
):
    x = lowpass_random(n_samples=n_samples, cutoff=cutoff, rng=seed, normalize=True)\

    #Collect means for plot
    means = []

    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(wspace=0.01, hspace=0.01, left=0.08, right=.98, top=.92)
    ax = plt.subplot2grid((rows, cols), (0, 0))
    for iter in range(0, len(a_list)):

        a = a_list[iter]
        a_not = 1 - a

        xe = pid_encode(x, kp=a, kd=a_not)  # Encode data to self.a*x + self.ki*self.s + self.a_not*(x-self.xp)
        h = Herder()
        xc = [h(xet) for xet in xe]  # round all the x's
        xd = pid_decode(xc, kp=a, kd=a_not)  # decode the xs, 1./float(a + ki + a_not) if (a + ki + a_not)>0 else np.inf

        means.append(np.abs(x - xd).mean())
        # FLIP COMMENTS FROM HERE BELOW TO GET THE OTHER PLOT

        this_ax = plt.subplot2grid((rows, cols), (iter // cols, iter % cols), sharex=ax, sharey=ax)
        plt.plot(xd, color='C1', label='$\hat x_t$')



        plt.text(.01, .01, '$\left<|x_t-\hat x_t|\\right>_t={:.2g}, \;\;\;  N={}$'.format(np.abs(x - xd).mean(),
                                                                                          int(np.sum(np.abs(xc)))),
                 ha='left', va='bottom', transform=this_ax.transAxes,
                 bbox=dict(boxstyle='square', facecolor='w', edgecolor='none', alpha=0.8, pad=0.0))

        if s_as_triangles:
            up_spikes = np.nonzero(xc > 0)[0]
            down_spikes = np.nonzero(xc < 0)[0]
            plt.plot(up_spikes, np.zeros(up_spikes.shape), '^', color='k', label='$s_t^+$')
            plt.plot(down_spikes, np.zeros(down_spikes.shape), 'v', color='r', label='$s_t^-$')
        else:
            plt.plot(xc, color='k', label='$s_t$')

        #
        plt.plot(x, color='C0', label='$x_t$')
        plt.grid()
        plt.xlabel('$a={}$'.format(a))
        #
        gs1 = gridspec.GridSpec(iter // cols, iter % cols)
        gs1.update(hspace=0.33)
    #
    ax.set_xlim(0, n_samples)
    ax.set_ylim(np.min(x) * 1.1, np.max(x) * 1.1)
    handles, labels = plt.gca().get_legend_handles_labels()
    #
    plt.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure,
               ncol=len(handles[::-1]), loc='upper right')
    #
    # plt.ylabel('<|x^_t- x_t|>.mean()')
    # plt.xlabel('a')
    # plt.plot(a_list, means, 'r')
    # plt.show()
    plt.show()


if __name__ == '__main__':
    demo_visualize_k_effects()
