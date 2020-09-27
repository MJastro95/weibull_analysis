import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import get_cmap

def dist_cdf(dist, edges):
    delta = edges[1:] - edges[:-1]

    return np.cumsum(dist*delta)

if __name__=="__main__":

    data = np.load("wb_onsala_post.npy", allow_pickle=True)
    post = data[0]
    k = data[1]
    r = data[2]

    kstart = k[0]
    kstop = k[-1]
    rstart = np.log10(r[0])
    rstop = np.log10(r[-1])

    k_cent = 0.5*(k[1:] + k[:-1])
    r_cent = 10**(0.5*(np.log10(r[1:]) + np.log10(r[:-1])))


    km, rm = np.meshgrid(k_cent, r_cent)



    k_delta = k[1:] - k[:-1]
    r_delta = r[1:] - r[:-1]

    kd_mesh, rd_mesh = np.meshgrid(k_delta, r_delta)


    # find marginal distribution of k
    k_marg = np.sum(post*rd_mesh, axis=0)

    # calculate cumulative distribution function of k
    k_c = dist_cdf(k_marg, k)

    # k median
    k_med = k[np.where(k_c>=0.5)[0][0]]

    # find marginal distribution of r
    r_marg = np.sum(post*kd_mesh, axis=1)

    # calculate cumulative distribution function of r
    r_c = dist_cdf(r_marg, r)

    # r median
    r_med = np.log10(r[np.where(r_c>=0.5)[0][0]])

    # calculate confidence intervals from cdf's
    k_99low = k_cent[np.where(k_c>=0.005)[0][0]]
    k_99high = k_cent[np.where(k_c>= 0.995)[0][0]]

    r_99low = np.log10(r_cent[np.where(r_c>=0.005)[0][0]])
    r_99high = np.log10(r_cent[np.where(r_c>=0.995)[0][0]])

    k_68low = k_cent[np.where(k_c>=0.16)[0][0]]
    k_68high = k_cent[np.where(k_c>=0.84)[0][0]]


    r_68low = np.log10(r_cent[np.where(r_c>=0.16)[0][0]])
    r_68high = np.log10(r_cent[np.where(r_c>=0.84)[0][0]])


    k_95low = k_cent[np.where(k_c>=0.025)[0][0]]
    k_95high = k_cent[np.where(k_c>=0.975)[0][0]]

    r_95low = np.log10(r_cent[np.where(r_c>=0.025)[0][0]])
    r_95high = np.log10(r_cent[np.where(r_c>=0.975)[0][0]])

    # calculate modes of marginal distribution
    k_mode = k_cent[np.where(k_marg==np.max(k_marg))[0][0]]
    r_mode = np.log10(r_cent[np.where(r_marg==np.max(r_marg))[0][0]])


    # create array of points (in units of max of distribution) to find confidence regions
    points = np.linspace(1, 0.0001, 1000)


    conf_regions = []
    flag68 = False
    flag95 = False
    flag99 = False
    # find 68%, 95%, and 99% confidence regions of joint distribution
    for i, point in enumerate(points):
        where = np.where(post>=point*np.max(post))

        val = np.sum(post[where]*kd_mesh[where]*rd_mesh[where])
        if (val >= 0.68) and not flag68:

            #refine point corresponding to confidence region
            for num in np.linspace(points[i-1], points[i]):
                where = np.where(post>=num*np.max(post)) 
                newval = np.sum(post[where]*kd_mesh[where]*rd_mesh[where])
                if (newval >= 0.68):
                    print("yes 68")
                    conf_regions.append(num*np.max(post))
                    flag68 = True
                    break
        elif (val>=0.95) and not flag95:


            #refine point corresponding to confidence region
            for num in np.linspace(points[i-1], points[i]):
                where = np.where(post>=num*np.max(post)) 
                newval = np.sum(post[where]*kd_mesh[where]*rd_mesh[where])
                if (newval >= 0.95):
                    print("yes 95")
                    conf_regions.append(num*np.max(post))
                    flag95 = True
                    break
        elif (val>=0.99) and not flag99:


            #refine point corresponding to confidence region
            for num in np.linspace(points[i-1], points[i]):
                where = np.where(post>=num*np.max(post)) 
                newval = np.sum(post[where]*kd_mesh[where]*rd_mesh[where])
                if (newval >= 0.99):
                    print("yes 99")
                    conf_regions.append(num*np.max(post))
                    flag99 = True
                    break
            break

    conf_regions = np.array(conf_regions)

    conf_regions = conf_regions[::-1]

    for conf in conf_regions:
        where = np.where(post>=conf)
        print(np.sum(post[where]*kd_mesh[where]*rd_mesh[where]))

    # create figure
    fig = plt.figure(figsize=(6, 4.5), dpi=100)
    ax1 = fig.add_axes([0.1, 0.1, 0.65, 0.65])
    ax2 = fig.add_axes([0.1, 0.775, 0.65, 0.15])
    ax3 = fig.add_axes([0.775, 0.1, 0.15, 0.65])

    cmap = get_cmap('Blues')
    color = cmap(0.8)

    low = (color[0], color[1], color[2], 0.5)
    mid = (color[0], color[1], color[2], 0.7)
    high = (color[0], color[1], color[2], 0.9)

    ax1.contourf(km, np.log10(rm), post, 
                levels=conf_regions, extend='max',colors=[low, mid, high])

    where_max = np.where(post==np.max(post))

    ax1.scatter(km[where_max], np.log10(rm[where_max]), s=40, marker="+", color="r")

    print("The mode is: k={:0.2f}, r={:0.4f}".format(km[where_max][0], rm[where_max][0]))
    ax1.set_ylim(-4, 2.5)
    ax1.set_xlim(kstart, 1)
    ax1.set_ylabel("log(r) (day$^{-1}$)")
    ax1.set_xlabel("k")


    ax2.plot(k_cent, k_marg)
    ax2.axvspan(k_99low, k_99high, alpha=0.4, color='navajowhite')
    ax2.axvspan(k_68low, k_68high, alpha=0.9, color='darkorange')
    ax2.axvspan(k_95low, k_95high, alpha=0.8, color='navajowhite')
    ax2.set_ylabel("$\mathcal{P}(k)$")
    ax2.set_xlim(kstart, 1)
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])


    print("95% confidence interval for k: {:0.2f}-{:0.2f}".format(k_95low, k_95high))
    print("68% confidence interval for r: {:0.2f}-{:0.2f}".format(10**r_68low, 10**r_68high))

    ax3.plot(r_marg, np.log10(r_cent))
    ax3.axhspan(r_99low, r_99high, alpha=0.4, color='navajowhite')
    ax3.axhspan(r_68low, r_68high, alpha=0.9, color='darkorange')
    ax3.axhspan(r_95low, r_95high, alpha=0.8, color='navajowhite')
    ax3.set_ylim(-4, 2.5)
    ax3.set_xticklabels([])
    ax3.set_xlabel("$\mathcal{P}(r)$")
    ax3.set_yticklabels([])

    plt.savefig('post_dens')








        