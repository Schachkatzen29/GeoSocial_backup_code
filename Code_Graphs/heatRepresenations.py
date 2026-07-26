##loading stuff 
import pandas as pd, numpy as np, seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

# exact path, space and parentheses included, inside the quotes
path = "/content/processed_maricopa_dailymax_2019_CLEAN (1).csv"
d = pd.read_csv(path, dtype={"GEOID": str}, parse_dates=["local_date"])
idx = {"Heat Index":"HI_F", "WBGT":"WBGT_clean_F", "UTCI":"UTCI_F"}
sub = d[list(idx.values())].dropna().rename(columns={v:k for k,v in idx.items()})
print("Loaded:", d.shape, "| rows for plots:", sub.shape)
plt.rcParams.update({"figure.dpi":120, "font.family":"DejaVu Sans", "axes.grid":False})


#Spearman's correlation matrix for different heat indices
corr = sub.corr(method="spearman")

fig, ax = plt.subplots(figsize=(5.6,4.6))
sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdYlBu_r", vmin=0.75, vmax=1.0,
            square=True, linewidths=2.5, linecolor="white",
            cbar_kws={"label":"Spearman ρ (rank correlation)"},
            annot_kws={"size":14, "weight":"bold"}, ax=ax)
ax.set_title("How similarly do the three heat-stress indices\nrank Phoenix summer days?  (daily max, May–Sept 2019)",
             fontsize=11, pad=14)
plt.xticks(rotation=0); plt.yticks(rotation=0)
plt.tight_layout(); plt.savefig("jobA_heatmap.png", dpi=200, bbox_inches="tight")
plt.show()


#spearman's p values and kendall's tau paired values
rows = []
for an, a in idx.items():
    for bn, b in idx.items():
        if an < bn:
            s = d[[a, b]].dropna()
            rho = spearmanr(s[a], s[b])[0]; tau = kendalltau(s[a], s[b])[0]
            rows.append({"Index pair": f"{an} vs {bn}", "Spearman ρ": rho,
                         "Kendall τ": tau, "Days disagree (%)": (1-tau)/2*100})
tbl = pd.DataFrame(rows)
(tbl.style
   .background_gradient(subset=["Spearman ρ"], cmap="RdYlGn", vmin=0.75, vmax=1.0)
   .background_gradient(subset=["Days disagree (%)"], cmap="Reds", vmin=0, vmax=20)
   .format({"Spearman ρ":"{:.3f}", "Kendall τ":"{:.3f}", "Days disagree (%)":"{:.1f}"})
   .set_caption("Job A — Heat-stress index divergence, Phoenix MSA summer 2019")
   .set_properties(**{"font-size":"12pt","text-align":"center"}))



#temporal graphs heat stress index vs time over 153 day period. adjusted wbgt values fixed by winsoring 
g0 = d["GEOID"].iloc[0]
one = d[d["GEOID"]==g0].sort_values("local_date")

fig, ax = plt.subplots(figsize=(12, 4.4))
# raw WBGT faint (shows the artifacts that were removed)
ax.plot(one["local_date"], one["WBGT_max_F"], color="#2c7fb8", lw=0.8, alpha=0.35,
        label="WBGT (raw, incl. artifacts)")
# cleaned lines solid
for name, c, col in [("Heat Index","HI_F","#c1440e"),
                     ("WBGT (cleaned)","WBGT_clean_F","#2c7fb8"),
                     ("UTCI","UTCI_F","#2ca25f")]:
    ax.plot(one["local_date"], one[c], label=name, lw=1.6, color=col)
ax.axvspan(pd.Timestamp("2019-07-01"), pd.Timestamp("2019-09-01"), alpha=0.07, color="navy")
ax.text(pd.Timestamp("2019-07-25"), ax.get_ylim()[1]*0.97, "monsoon (Jul–Aug)",
        ha="center", fontsize=9, color="navy")
ax.set_ylabel("Daily max (°F)"); ax.legend(ncol=2, fontsize=9, loc="lower center")
ax.set_title(f"Three heat-stress indices, one Phoenix tract, summer 2019\n(gaps in cleaned WBGT = removed radiation artifacts)",
             fontsize=11)
plt.tight_layout(); plt.savefig("timeseries_annotated.png", dpi=200, bbox_inches="tight"); plt.show()


#more loading stuff
!pip install pygris geopandas mapclassify --quiet
import geopandas as gpd, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pygris import tracts

# Maricopa County (04=AZ, 013=Maricopa), 2010 vintage to match the heat data
az = tracts(state="04", county="013", year=2019, cache=True)   # TIGER/Line polygons
az["GEOID"] = az["GEOID"].astype(str)

# Per-tract summaries of the daily-max heat (mean & 95th pct across the summer)
agg = (d.groupby("GEOID")
         .agg(HI_mean=("HI_F","mean"),   HI_p95=("HI_F", lambda x: x.quantile(.95)),
              WBGT_mean=("WBGT_clean_F","mean"), WBGT_p95=("WBGT_clean_F", lambda x: x.quantile(.95)),
              UTCI_mean=("UTCI_F","mean"), UTCI_p95=("UTCI_F", lambda x: x.quantile(.95)),
              temp_mean=("airtemp_F","mean"))
         .reset_index())

gdf = az.merge(agg, on="GEOID", how="left")
print("Tracts with geometry:", len(az), "| matched to heat:", gdf["HI_mean"].notna().sum())


#distributions of each heat stress index and their scatterplot graphed with each other 
plot_sub = sub.sample(min(8000, len(sub)), random_state=1)

g = sns.pairplot(plot_sub, corner=False,          # <-- full grid, HI on both axes
                 plot_kws={"s":6, "alpha":0.2, "color":"#c1440e", "edgecolor":"none"},
                 diag_kws={"color":"#3bb273", "bins":45})

# identity line on every off-diagonal panel
for i in range(3):
    for j in range(3):
        if i != j:
            ax = g.axes[i][j]
            lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
            hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)

g.fig.suptitle("Do the indices agree day-to-day?  (points on dashed line = perfect agreement)",
               y=1.02, fontsize=12)
plt.tight_layout()
plt.savefig("jobA_scatter_full.png", dpi=200, bbox_inches="tight")
plt.show()



#summer median daily-max of each census tract. spatial representation
K = 4
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (col, name) in zip(axes, [("HI_med","Heat Index"),("WBGT_med","WBGT"),("UTCI_med","UTCI")]):
    gdf.plot(column=col, ax=ax, cmap="YlOrRd", scheme="quantiles", k=K,
             edgecolor="white", linewidth=0.1, legend=True,
             legend_kwds={"title":f"{name} (°F)", "loc":"lower left", "fontsize":8},
             missing_kwds={"color":"lightgrey"})
    ax.set_title(name, fontsize=13); ax.axis("off")
fig.suptitle(f"Summer-median daily-max heat by tract — Maricopa County 2019  (quantile, {K} classes)",
             fontsize=14, y=1.03)
plt.tight_layout(); plt.savefig("map_three_median.png", dpi=200, bbox_inches="tight"); plt.show()




#index representations by month sorted 
import seaborn as sns, matplotlib.pyplot as plt, pandas as pd

plot_df = d[["HI_F","WBGT_clean_F","UTCI_F","local_date"]].dropna().copy()
plot_df["Month"] = pd.to_datetime(plot_df["local_date"]).dt.strftime("%b")
plot_df = plot_df.rename(columns={"HI_F":"Heat Index","WBGT_clean_F":"WBGT","UTCI_F":"UTCI"})

month_order = ["May","Jun","Jul","Aug","Sep"]
plot_df["Month"] = pd.Categorical(plot_df["Month"], categories=month_order, ordered=True)
samp = plot_df.sample(min(9000, len(plot_df)), random_state=1)

# high-contrast distinct colors (not a faded gradient)
palette = {"May":"#1f77b4","Jun":"#2ca02c","Jul":"#ff7f0e","Aug":"#d62728","Sep":"#9467bd"}

g = sns.pairplot(samp, vars=["Heat Index","WBGT","UTCI"], hue="Month",
                 hue_order=month_order, corner=False, palette=palette,
                 plot_kws={"s":14, "alpha":0.6, "edgecolor":"none"},   # darker, bigger dots
                 diag_kind="kde")

for i in range(3):
    for j in range(3):
        if i != j:
            ax = g.axes[i][j]
            lo = min(ax.get_xlim()[0], ax.get_ylim()[0]); hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([lo,hi],[lo,hi],"k--",lw=1,alpha=0.5)

# make the legend big and readable
g._legend.set_title("Month")
for handle in g._legend.legend_handles:
    handle.set_markersize(12); handle.set_alpha(1.0)
plt.setp(g._legend.get_texts(), fontsize=12)
plt.setp(g._legend.get_title(), fontsize=13)

g.fig.suptitle("Index relationships by month — do dry (May–Jun) and monsoon (Jul–Aug) separate?",
               y=1.02, fontsize=12)
plt.savefig("scatter_by_month.png", dpi=200, bbox_inches="tight"); plt.show()



#index divergence by tract 
from scipy.stats import spearmanr
def tract_divergence(sub):
    s = sub[["HI_F","WBGT_clean_F","UTCI_F"]].dropna()
    if len(s) < 10 or s["WBGT_clean_F"].nunique() < 3: return np.nan
    return 1 - np.nanmean([spearmanr(s["HI_F"],s["WBGT_clean_F"])[0],
                           spearmanr(s["HI_F"],s["UTCI_F"])[0],
                           spearmanr(s["WBGT_clean_F"],s["UTCI_F"])[0]])
div = d.groupby("GEOID").apply(tract_divergence).rename("divergence").reset_index()
gdf_div = az.merge(div, on="GEOID", how="left")

fig, ax = plt.subplots(figsize=(10,10))
gdf_div.plot(column="divergence", ax=ax, cmap="YlOrRd", scheme="quantiles", k=5,
             edgecolor="white", linewidth=0.15, legend=True,
             legend_kwds={"title":"Index divergence\n(1 − mean pairwise ρ)","loc":"lower right","fontsize":8},
             missing_kwds={"color":"lightgrey"})
ax.set_title("Where does the choice of heat index matter most?\nPer-tract index divergence, Phoenix summer 2019", fontsize=13)
ax.axis("off"); plt.tight_layout(); plt.savefig("map_divergence.png", dpi=200, bbox_inches="tight"); plt.show()






