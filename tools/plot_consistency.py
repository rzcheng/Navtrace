"""Full-scale scientific plots of measured M3 outputs (predeclared lowest seed)."""
from tools.plot_results import plt  # Chooses headless backend and writable cache first.
from matplotlib.patches import Ellipse
import numpy as np
from scipy.stats import chi2


def plot_case(output, scenario, seed, series):
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    colors={'ekf':'#2877b5','gated':'#d47816'}
    for method, (times, states, covariance, truth, stats, events) in series.items():
        t=times/1e6; color=colors[method]
        error=np.linalg.norm(states[:,:2]-truth[:,:2],axis=1)
        major=np.sqrt(chi2.ppf(.95,2)*np.linalg.eigvalsh(covariance[:,:2,:2])[:,-1])
        axes[0,0].plot(t,error,color=color,label=f'{method} error')
        axes[0,0].plot(t,major,color=color,ls='--',alpha=.7,label=f'{method} 95% ellipse major radius')
        for ax,name in [(axes[0,1],'heading'),(axes[1,0],'bias')]:
            ax.plot(t,stats[name+'_error'],color=color,label=method)
            ax.fill_between(t,-1.96*stats[name+'_std'],1.96*stats[name+'_std'],color=color,alpha=.13)
        axes[1,1].plot(t,stats['position_nees'],color=color,label=method)
        axes[2,0].plot(t,stats['rate_bias_correlation'],color=color,label=method)
        et=np.array([r['timestamp_us']/1e6 for r in events]); nis=np.array([r['nis'] for r in events])
        # Break the connecting line across missing GPS; no innovation exists there.
        breaks=np.flatnonzero(np.diff(et)>.200001)+1
        plot_t=np.insert(et,breaks,np.nan); plot_nis=np.insert(nis,breaks,np.nan)
        axes[2,1].plot(plot_t,plot_nis,color=color,lw=.8,label=f'{method}, all available fixes')
        rejected=np.array([r['status']=='rejected' for r in events])
        axes[2,1].scatter(et[rejected],nis[rejected],s=12,color=color,marker='x')
    axes[1,1].axhline(chi2.ppf(.95,2),color='black',ls=':',label='nominal 95%, d=2')
    axes[2,1].axhline(9.210340371976184,color='black',ls=':')
    axes[1,1].set_yscale('symlog',linthresh=1);axes[1,1].set_ylim(bottom=0)
    axes[2,1].set_yscale('symlog',linthresh=1);axes[2,1].set_ylim(bottom=0)
    titles=['Position error / ellipse radius (m)', 'Yaw error and ±1.96 std (rad)',
            'Bias error and ±1.96 std (rad/s)','Position NEES (marginal P)',
            'Yaw-rate / bias correlation','Pre-decision GPS NIS; x = rejected']
    for ax,title in zip(axes.flat,titles):
        ax.set_title(title,fontsize=10);ax.grid(alpha=.2)
        if scenario['outage_us']:
            a,b=scenario['outage_us'];ax.axvspan(a/1e6,b/1e6,color='gold',alpha=.15)
        for fault in scenario['offsets']:
            ax.axvspan(fault['start_us']/1e6,fault['end_us']/1e6,color='purple',alpha=.1)
    axes[0,0].legend(fontsize=7);axes[0,1].legend(fontsize=8)
    axes[2,0].set_xlabel('Time (s)');axes[2,1].set_xlabel('Time (s)')
    fig.suptitle(f"{scenario['id']} · predefined illustrative seed {seed}\nYellow: GPS absent; purple: injected offset. Nominal uncertainty references, not calibration proof.")
    fig.tight_layout();fig.savefig(output/f"{scenario['id']}-uncertainty.png",dpi=150);plt.close(fig)
    if scenario['outage_us']:
        select=[scenario['outage_us'][1]-10000,scenario['outage_us'][1],scenario['recovery_windows'][0]['reference_us'],60000000]
        fig,axes=plt.subplots(2,4,figsize=(13,6))
        for row,(method,(times,states,covariance,truth,stats,events)) in enumerate(series.items()):
            for ax,timestamp in zip(axes[row],select):
                index=int(np.flatnonzero(times==timestamp)[0]); p=covariance[index,:2,:2]
                values,vectors=np.linalg.eigh(p); radii=np.sqrt(chi2.ppf(.95,2)*values)
                angle=np.degrees(np.arctan2(vectors[1,1],vectors[0,1]))
                ax.add_patch(Ellipse((0,0),2*radii[1],2*radii[0],angle=angle,facecolor=colors[method],alpha=.2))
                difference=truth[index,:2]-states[index,:2]
                ax.plot([0,difference[0]],[0,difference[1]],'r.-',lw=1)
                # Include the full ellipse and observed error, with equal metric axes.
                bound=max(float(radii[-1]),float(np.abs(difference).max()),.05)*1.15
                ax.set_xlim(-bound,bound);ax.set_ylim(-bound,bound);ax.set_aspect('equal');ax.grid(alpha=.2)
                ax.set_title(f'{method} at {timestamp/1e6:g}s',fontsize=9)
                ax.set_xlabel('World x error (m)')
            axes[row,0].set_ylabel('World y error (m)')
        fig.suptitle(f"{scenario['id']} · seed {seed}: 95% marginal position ellipses\nCentered on estimate; red endpoint is truth minus estimate; each panel has its own full scale.")
        fig.tight_layout();fig.savefig(output/f"{scenario['id']}-ellipses.png",dpi=150);plt.close(fig)
