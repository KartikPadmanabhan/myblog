
# USING METHOD OF MOMENTS TO PREDICT DATABASE PERFORMANCE METRICS

## Introduction

This article exploits statistical technique method of moments (MOM) and generalized method of moments (GMM)  to estimate database statistics from the past data. Both GMM and MOM's are parametric  unsupervised form of learning. I am restricting this to Oracle Database in this example, but it can be extended to any database system.

We initially select a distribution to model our random variable. Our random variables in database statistical parameters could be any statistic that we collect say in AWR report. To start our analysis, we collected AWR reports over several months. We then scraped these data and stored our metrics in csv formats. I have presented some csv files corresponding to these statistics in my github page.


## Parameters Of Interest

Lets say we want to use the method of moments to predict the following AWR statistics for the database Load Profile in AWR statistics. This is given in this section.


    from IPython.display import Image
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats.kde import gaussian_kde
    from plotly import tools
    import scipy.stats as scs
    import plotly.plotly as py
    from plotly.graph_objs import *
    import plotly.plotly as py
    from plotly.graph_objs import *
    import numpy as np


    Image(filename='images/load_profile.png')




![png](method_of_moments_files/method_of_moments_6_0.png)



Reading our historic load profile data in the dataframe.


    df=pd.read_csv('data/TMP_AWR_LOAD_PROFILE_AGG.csv', sep='|',parse_dates=True)
    df.head()




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>site_id</th>
      <th>config_id</th>
      <th>run_id</th>
      <th>stat_id</th>
      <th>stat_name</th>
      <th>stat_per_sec</th>
      <th>stat_per_txn</th>
      <th>stat_per_exec</th>
      <th>stat_per_call</th>
      <th>start_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>37</td>
      <td>1</td>
      <td>1</td>
      <td>265418</td>
      <td>Global Cache blocks received</td>
      <td>4467.21</td>
      <td>16.36</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>03-OCT-14 08.00.00.000000 AM</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
      <td>1</td>
      <td>1</td>
      <td>265427</td>
      <td>Transactions</td>
      <td>274.49</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>03-OCT-14 08.00.00.000000 AM</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>1</td>
      <td>1</td>
      <td>265410</td>
      <td>Logical read (blocks)</td>
      <td>291880.19</td>
      <td>1064.87</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>03-OCT-14 08.00.00.000000 AM</td>
    </tr>
    <tr>
      <th>3</th>
      <td>37</td>
      <td>1</td>
      <td>1</td>
      <td>265419</td>
      <td>Global Cache blocks served</td>
      <td>4467.02</td>
      <td>16.34</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>03-OCT-14 08.00.00.000000 AM</td>
    </tr>
    <tr>
      <th>4</th>
      <td>37</td>
      <td>1</td>
      <td>1</td>
      <td>265421</td>
      <td>Parses (SQL)</td>
      <td>1131.04</td>
      <td>4.14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>03-OCT-14 08.00.00.000000 AM</td>
    </tr>
  </tbody>
</table>
</div>



As we can we have the data for several statistics viz. Global Cache blocks received, Transactions, Logical read (blocks), Global Cache blocks served, etc from the AWR load profile start from date 03-OCT-14  until 02-NOV-15.
I am interested in forecasting only stat_per_sec information on this AWR Load profile, so using pivot function in pandas we get the following reconstructed dataframe.      


    df=df.pivot_table(index='start_time', columns='stat_name', values='stat_per_sec')
    df.head()




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>stat_name</th>
      <th>Background CPU(s)</th>
      <th>Block changes</th>
      <th>DB CPU(s)</th>
      <th>DB Time(s)</th>
      <th>Executes (SQL)</th>
      <th>Global Cache blocks received</th>
      <th>Global Cache blocks served</th>
      <th>Hard parses (SQL)</th>
      <th>IM scan rows</th>
      <th>Logical read (blocks)</th>
      <th>...</th>
      <th>Read IO (MB)</th>
      <th>Read IO requests</th>
      <th>Redo size (bytes)</th>
      <th>Rollbacks</th>
      <th>SQL Work Area (MB)</th>
      <th>Session Logical Read IM</th>
      <th>Transactions</th>
      <th>User calls</th>
      <th>Write IO (MB)</th>
      <th>Write IO requests</th>
    </tr>
    <tr>
      <th>start_time</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>01-APR-15 08.00.00.000000 AM</th>
      <td>2.34</td>
      <td>7171.66</td>
      <td>8.34</td>
      <td>21.33</td>
      <td>7295.85</td>
      <td>4067.55</td>
      <td>4067.66</td>
      <td>0.53</td>
      <td>0</td>
      <td>368308.41</td>
      <td>...</td>
      <td>146.02</td>
      <td>12323.87</td>
      <td>2520626.66</td>
      <td>204.03</td>
      <td>471.86</td>
      <td>NaN</td>
      <td>342.13</td>
      <td>3172.36</td>
      <td>7.38</td>
      <td>791.23</td>
    </tr>
    <tr>
      <th>01-APR-15 10.00.00.000000 PM</th>
      <td>1.88</td>
      <td>5829.02</td>
      <td>5.75</td>
      <td>14.98</td>
      <td>4943.24</td>
      <td>3371.89</td>
      <td>3374.17</td>
      <td>0.41</td>
      <td>0</td>
      <td>274450.76</td>
      <td>...</td>
      <td>237.88</td>
      <td>9459.97</td>
      <td>1708287.87</td>
      <td>143.23</td>
      <td>282.08</td>
      <td>NaN</td>
      <td>243.01</td>
      <td>2288.04</td>
      <td>5.59</td>
      <td>603.88</td>
    </tr>
    <tr>
      <th>01-JAN-15 08.00.00.000000 AM</th>
      <td>NaN</td>
      <td>1746.34</td>
      <td>1.80</td>
      <td>4.07</td>
      <td>2277.13</td>
      <td>1618.04</td>
      <td>1617.80</td>
      <td>0.30</td>
      <td>NaN</td>
      <td>119422.36</td>
      <td>...</td>
      <td>160.56</td>
      <td>3976.52</td>
      <td>577855.43</td>
      <td>74.32</td>
      <td>113.44</td>
      <td>NaN</td>
      <td>139.59</td>
      <td>970.47</td>
      <td>1.61</td>
      <td>162.10</td>
    </tr>
    <tr>
      <th>01-JUL-15 08.00.00.000000 AM</th>
      <td>2.07</td>
      <td>6530.64</td>
      <td>6.92</td>
      <td>18.46</td>
      <td>6485.98</td>
      <td>4164.01</td>
      <td>4164.06</td>
      <td>1.67</td>
      <td>0</td>
      <td>306467.96</td>
      <td>...</td>
      <td>155.87</td>
      <td>9908.58</td>
      <td>2263412.36</td>
      <td>198.20</td>
      <td>424.66</td>
      <td>NaN</td>
      <td>334.05</td>
      <td>3064.53</td>
      <td>6.76</td>
      <td>721.93</td>
    </tr>
    <tr>
      <th>01-JUL-15 10.00.00.000000 PM</th>
      <td>1.80</td>
      <td>5853.14</td>
      <td>5.19</td>
      <td>14.20</td>
      <td>4787.97</td>
      <td>3469.74</td>
      <td>3469.68</td>
      <td>0.94</td>
      <td>0</td>
      <td>248284.74</td>
      <td>...</td>
      <td>192.59</td>
      <td>8087.20</td>
      <td>1731536.10</td>
      <td>151.27</td>
      <td>288.96</td>
      <td>NaN</td>
      <td>259.02</td>
      <td>2370.28</td>
      <td>5.70</td>
      <td>606.48</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




    print (df.columns)

    Index(['Background CPU(s)', 'Block changes', 'DB CPU(s)', 'DB Time(s)',
           'Executes (SQL)', 'Global Cache blocks received',
           'Global Cache blocks served', 'Hard parses (SQL)', 'IM scan rows',
           'Logical read (blocks)', 'Logons', 'Parses (SQL)',
           'Physical read (blocks)', 'Physical write (blocks)', 'Read IO (MB)',
           'Read IO requests', 'Redo size (bytes)', 'Rollbacks',
           'SQL Work Area (MB)', 'Session Logical Read IM', 'Transactions',
           'User calls', 'Write IO (MB)', 'Write IO requests'],
          dtype='object', name='stat_name')


## Distribution Selection

All of these database statistics are continous random variables, so we model using some continous distributions.
scipy offers several continous distributions for the purpose of modelling, viz Gamma, Normal, Anglit, Arcsine, Beta, Cauchy etc.

Once we have the estimated model, we can then use the Kolmogorov-Smirov test to evaluate your fit using scipy.stats.kstest. Lets start with the most popular gamma distribution.

Say I am interested in Background CPU(s) information, calculating the gamma distribution parameters alpha and beta.



    sample_mean=df['Background CPU(s)'].mean()
    sample_var=np.var(df['Background CPU(s)'], ddof=1)
    (sample_mean,sample_var)




    (2.1573536895674299, 0.2304741016253824)



Alphas and Betas of Gamma distribution:


    ###############
    #### Gamma ####
    ###############
    
    alpha = sample_mean**2 / sample_var
    beta = sample_mean / sample_var 
    alpha,beta




    (20.193917273426234, 9.3605037370925057)



## MOM Estimation

Using estimated parameters and plot the distribution on top of data


    gamma_rv = scs.gamma(a=alpha, scale=1/beta)


    # Get the probability for each value in the data
    x_vals = np.linspace(df['Background CPU(s)'].min(), df['Background CPU(s)'].max())
    gamma_p = gamma_rv.pdf(x_vals)


    df.head()




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>stat_name</th>
      <th>Background CPU(s)</th>
      <th>Block changes</th>
      <th>DB CPU(s)</th>
      <th>DB Time(s)</th>
      <th>Executes (SQL)</th>
      <th>Global Cache blocks received</th>
      <th>Global Cache blocks served</th>
      <th>Hard parses (SQL)</th>
      <th>IM scan rows</th>
      <th>Logical read (blocks)</th>
      <th>...</th>
      <th>Read IO (MB)</th>
      <th>Read IO requests</th>
      <th>Redo size (bytes)</th>
      <th>Rollbacks</th>
      <th>SQL Work Area (MB)</th>
      <th>Session Logical Read IM</th>
      <th>Transactions</th>
      <th>User calls</th>
      <th>Write IO (MB)</th>
      <th>Write IO requests</th>
    </tr>
    <tr>
      <th>start_time</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>01-APR-15 08.00.00.000000 AM</th>
      <td>2.34</td>
      <td>7171.66</td>
      <td>8.34</td>
      <td>21.33</td>
      <td>7295.85</td>
      <td>4067.55</td>
      <td>4067.66</td>
      <td>0.53</td>
      <td>0</td>
      <td>368308.41</td>
      <td>...</td>
      <td>146.02</td>
      <td>12323.87</td>
      <td>2520626.66</td>
      <td>204.03</td>
      <td>471.86</td>
      <td>NaN</td>
      <td>342.13</td>
      <td>3172.36</td>
      <td>7.38</td>
      <td>791.23</td>
    </tr>
    <tr>
      <th>01-APR-15 10.00.00.000000 PM</th>
      <td>1.88</td>
      <td>5829.02</td>
      <td>5.75</td>
      <td>14.98</td>
      <td>4943.24</td>
      <td>3371.89</td>
      <td>3374.17</td>
      <td>0.41</td>
      <td>0</td>
      <td>274450.76</td>
      <td>...</td>
      <td>237.88</td>
      <td>9459.97</td>
      <td>1708287.87</td>
      <td>143.23</td>
      <td>282.08</td>
      <td>NaN</td>
      <td>243.01</td>
      <td>2288.04</td>
      <td>5.59</td>
      <td>603.88</td>
    </tr>
    <tr>
      <th>01-JAN-15 08.00.00.000000 AM</th>
      <td>NaN</td>
      <td>1746.34</td>
      <td>1.80</td>
      <td>4.07</td>
      <td>2277.13</td>
      <td>1618.04</td>
      <td>1617.80</td>
      <td>0.30</td>
      <td>NaN</td>
      <td>119422.36</td>
      <td>...</td>
      <td>160.56</td>
      <td>3976.52</td>
      <td>577855.43</td>
      <td>74.32</td>
      <td>113.44</td>
      <td>NaN</td>
      <td>139.59</td>
      <td>970.47</td>
      <td>1.61</td>
      <td>162.10</td>
    </tr>
    <tr>
      <th>01-JUL-15 08.00.00.000000 AM</th>
      <td>2.07</td>
      <td>6530.64</td>
      <td>6.92</td>
      <td>18.46</td>
      <td>6485.98</td>
      <td>4164.01</td>
      <td>4164.06</td>
      <td>1.67</td>
      <td>0</td>
      <td>306467.96</td>
      <td>...</td>
      <td>155.87</td>
      <td>9908.58</td>
      <td>2263412.36</td>
      <td>198.20</td>
      <td>424.66</td>
      <td>NaN</td>
      <td>334.05</td>
      <td>3064.53</td>
      <td>6.76</td>
      <td>721.93</td>
    </tr>
    <tr>
      <th>01-JUL-15 10.00.00.000000 PM</th>
      <td>1.80</td>
      <td>5853.14</td>
      <td>5.19</td>
      <td>14.20</td>
      <td>4787.97</td>
      <td>3469.74</td>
      <td>3469.68</td>
      <td>0.94</td>
      <td>0</td>
      <td>248284.74</td>
      <td>...</td>
      <td>192.59</td>
      <td>8087.20</td>
      <td>1731536.10</td>
      <td>151.27</td>
      <td>288.96</td>
      <td>NaN</td>
      <td>259.02</td>
      <td>2370.28</td>
      <td>5.70</td>
      <td>606.48</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




    '''Use the estimated parameters and plot the distribution on top of data'''
    fig, ax = plt.subplots()
    # Plot those values on top of the real data
    ax = df['Background CPU(s)'].hist(bins=20, normed=1, edgecolor='none', figsize=(10, 7))
    ax.set_xlabel('Statistics')
    ax.set_ylabel('Probability Density')
    ax.set_title('Background CPU(s)')
    
    ax.plot(x_vals, gamma_p, color='r', label='Gamma', alpha=0.6)
    ax.legend()
    fig.savefig('images/backgroundcpu.png')
    #py.iplot_mpl(fig, filename='s6_log-scales')


![png](method_of_moments_files/method_of_moments_22_0.png)



    # Lets put everything in a function 
    # Define a function that plots distribution fitted to one month's of data
    def plot_mom(df, col):
        sample_mean=df[col].mean()
        sample_var=np.var(df[col], ddof=1)
        alpha = sample_mean**2 / sample_var
        beta = sample_mean / sample_var 
        gamma_rv = scs.gamma(a=alpha, scale=1/beta)
        # Get the probability for each value in the data
        x_vals = np.linspace(df[col].min(), df[col].max())
        gamma_p = gamma_rv.pdf(x_vals)
        fig, ax = plt.subplots()
        # Plot those values on top of the real data
        ax = df[col].hist(bins=20, normed=1, edgecolor='none')
        ax.set_xlabel('Statistics')
        ax.set_ylabel('Probability Density')
        ax.set_title(col)
        ax.plot(x_vals, gamma_p, color='r', label='Gamma', alpha=0.6)
        ax.legend()


    df.columns




    Index(['Background CPU(s)', 'Block changes', 'DB CPU(s)', 'DB Time(s)',
           'Executes (SQL)', 'Global Cache blocks received',
           'Global Cache blocks served', 'Hard parses (SQL)', 'IM scan rows',
           'Logical read (blocks)', 'Logons', 'Parses (SQL)',
           'Physical read (blocks)', 'Physical write (blocks)', 'Read IO (MB)',
           'Read IO requests', 'Redo size (bytes)', 'Rollbacks',
           'SQL Work Area (MB)', 'Session Logical Read IM', 'Transactions',
           'User calls', 'Write IO (MB)', 'Write IO requests'],
          dtype='object', name='stat_name')



Lets just pick the following important statistics to estimate:
	* 'Background CPU(s)'
	* 'DB CPU(s)',
	* 'Executes (SQL)',
	* 'IM scan rows',
	* 'Hard parses (SQL)',
	* 'Logical read (blocks)',
	* 'Parses (SQL)',
	* 'Physical read (blocks)',
	* 'Physical write (blocks)'


    columns_of_interest=['Background CPU(s)', 'DB CPU(s)', 'Executes (SQL)', 'User calls', 'Hard parses (SQL)', 'Logical read (blocks)', 'Parses (SQL)', 'Physical read (blocks)', 'Physical write (blocks)']


    for col in columns_of_interest:
        plot_mom(df, col)
        fig.savefig('images/'+col)


![png](method_of_moments_files/method_of_moments_27_0.png)



![png](method_of_moments_files/method_of_moments_27_1.png)



![png](method_of_moments_files/method_of_moments_27_2.png)



![png](method_of_moments_files/method_of_moments_27_3.png)



![png](method_of_moments_files/method_of_moments_27_4.png)



![png](method_of_moments_files/method_of_moments_27_5.png)



![png](method_of_moments_files/method_of_moments_27_6.png)



![png](method_of_moments_files/method_of_moments_27_7.png)



![png](method_of_moments_files/method_of_moments_27_8.png)



    df.head()




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>stat_name</th>
      <th>Background CPU(s)</th>
      <th>Block changes</th>
      <th>DB CPU(s)</th>
      <th>DB Time(s)</th>
      <th>Executes (SQL)</th>
      <th>Global Cache blocks received</th>
      <th>Global Cache blocks served</th>
      <th>Hard parses (SQL)</th>
      <th>IM scan rows</th>
      <th>Logical read (blocks)</th>
      <th>...</th>
      <th>Read IO (MB)</th>
      <th>Read IO requests</th>
      <th>Redo size (bytes)</th>
      <th>Rollbacks</th>
      <th>SQL Work Area (MB)</th>
      <th>Session Logical Read IM</th>
      <th>Transactions</th>
      <th>User calls</th>
      <th>Write IO (MB)</th>
      <th>Write IO requests</th>
    </tr>
    <tr>
      <th>start_time</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>01-APR-15 08.00.00.000000 AM</th>
      <td>2.34</td>
      <td>7171.66</td>
      <td>8.34</td>
      <td>21.33</td>
      <td>7295.85</td>
      <td>4067.55</td>
      <td>4067.66</td>
      <td>0.53</td>
      <td>0</td>
      <td>368308.41</td>
      <td>...</td>
      <td>146.02</td>
      <td>12323.87</td>
      <td>2520626.66</td>
      <td>204.03</td>
      <td>471.86</td>
      <td>NaN</td>
      <td>342.13</td>
      <td>3172.36</td>
      <td>7.38</td>
      <td>791.23</td>
    </tr>
    <tr>
      <th>01-APR-15 10.00.00.000000 PM</th>
      <td>1.88</td>
      <td>5829.02</td>
      <td>5.75</td>
      <td>14.98</td>
      <td>4943.24</td>
      <td>3371.89</td>
      <td>3374.17</td>
      <td>0.41</td>
      <td>0</td>
      <td>274450.76</td>
      <td>...</td>
      <td>237.88</td>
      <td>9459.97</td>
      <td>1708287.87</td>
      <td>143.23</td>
      <td>282.08</td>
      <td>NaN</td>
      <td>243.01</td>
      <td>2288.04</td>
      <td>5.59</td>
      <td>603.88</td>
    </tr>
    <tr>
      <th>01-JAN-15 08.00.00.000000 AM</th>
      <td>NaN</td>
      <td>1746.34</td>
      <td>1.80</td>
      <td>4.07</td>
      <td>2277.13</td>
      <td>1618.04</td>
      <td>1617.80</td>
      <td>0.30</td>
      <td>NaN</td>
      <td>119422.36</td>
      <td>...</td>
      <td>160.56</td>
      <td>3976.52</td>
      <td>577855.43</td>
      <td>74.32</td>
      <td>113.44</td>
      <td>NaN</td>
      <td>139.59</td>
      <td>970.47</td>
      <td>1.61</td>
      <td>162.10</td>
    </tr>
    <tr>
      <th>01-JUL-15 08.00.00.000000 AM</th>
      <td>2.07</td>
      <td>6530.64</td>
      <td>6.92</td>
      <td>18.46</td>
      <td>6485.98</td>
      <td>4164.01</td>
      <td>4164.06</td>
      <td>1.67</td>
      <td>0</td>
      <td>306467.96</td>
      <td>...</td>
      <td>155.87</td>
      <td>9908.58</td>
      <td>2263412.36</td>
      <td>198.20</td>
      <td>424.66</td>
      <td>NaN</td>
      <td>334.05</td>
      <td>3064.53</td>
      <td>6.76</td>
      <td>721.93</td>
    </tr>
    <tr>
      <th>01-JUL-15 10.00.00.000000 PM</th>
      <td>1.80</td>
      <td>5853.14</td>
      <td>5.19</td>
      <td>14.20</td>
      <td>4787.97</td>
      <td>3469.74</td>
      <td>3469.68</td>
      <td>0.94</td>
      <td>0</td>
      <td>248284.74</td>
      <td>...</td>
      <td>192.59</td>
      <td>8087.20</td>
      <td>1731536.10</td>
      <td>151.27</td>
      <td>288.96</td>
      <td>NaN</td>
      <td>259.02</td>
      <td>2370.28</td>
      <td>5.70</td>
      <td>606.48</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



This above function would generate the MOM estimate for any database statistic that we would want.
For example if we want to predict MOM's for "Global Cache blocks served", we just need to call this function:


    plot_mom(df,'Global Cache blocks received')
    fig.savefig('images/cache_blocks.png')


![png](method_of_moments_files/method_of_moments_30_0.png)


For "Rollbacks" example our estimation is:


    plot_mom(df,'Rollbacks')
    fig.savefig('images/rollbacks.png')


![png](method_of_moments_files/method_of_moments_32_0.png)


## Conclusion:

As we can we our estimations (plotted in red) are pretty good estimation of our actual values that we observe in our real outputs. This concludes our discussion on MOM estimators on database statistics.
