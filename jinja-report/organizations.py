#!/usr/bin/env python3

import os
import pandas
from datetime import datetime
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import cm
import numpy

import plot

register_matplotlib_converters()


def load_data(workingdir):

    # load the data
    path = os.path.join(workingdir, 'users.pkl')
    df = pandas.read_pickle(path)

    # convert dates
    df['date'] = pandas.to_datetime(df.usr_created_date).dt.normalize()
    df.usr_created_date = pandas.to_datetime(df.usr_created_date) \
                                .dt.normalize()
    df.usr_last_login_date = pandas.to_datetime(df.usr_last_login_date) \
                                   .dt.normalize()
    df.report_date = pandas.to_datetime(df.report_date).dt.normalize()

    # add another date column and make it the index
    df['Date'] = df['date']

    # change the index to timestamp
    df.set_index(['Date'], inplace=True)

    return df


def subset_by_date(dat, st, et):

    if type(dat) == pandas.DataFrame:

        # select dates between start/end range
        mask = (dat.date >= st) & (dat.date < et)
        dat = dat.loc[mask]
        return dat

    elif type(dat) == pandas.Series:

        # select dates between start/end range
        mask = (dat.index >= st) & (dat.index < et)
        return dat.loc[mask]


def total_table(input_directory='.',
                start_time=datetime(2000, 1, 1),
                end_time=datetime(2030, 1, 1),
                aggregation='Q',
                **kwargs):
   
    print('--> calculating total distinct organizations - table')

    # load the data based on working directory and subset it if necessary
    df = load_data(input_directory)
    df = subset_by_date(df, start_time, end_time)

    # drop duplicates (except the first occurrence)
    # df = df.drop_duplicates(subset='usr_organization', keep='first')

    df = df.filter(['usr_organization'], axis=1)

    df.mask(df.eq('None')).dropna()

    # create columns for each org
    for uo in df.usr_organization.unique():
        df[uo] = numpy.where(df.usr_organization == uo, 1, 0)
    
    # remove the usr_organization column since its been divided into
    # individual columns
    df.drop('usr_organization', axis=1, inplace=True)

    df = df[df.columns[df.sum()> kwargs.get('users_threshold', 25)]]

    # group and cumsum
    # df = df.sort_index()
    # ds = df.groupby(pandas.Grouper(freq=aggregation)) \
    #         .usr_organization.nunique().cumsum()
    df = df.groupby(pandas.Grouper(freq=aggregation)).sum()

    # # drop unnecessary columns
    # df = df.filter(['Date', 'begin_session', 'login', 'delete',
    #                 'create', 'download', 'app_launch'], axis=1)

    # # rename columns
    # df = df.rename(columns={'begin_session': 'Begin\nSession',
    #                         'login': 'Login',
    #                         'delete': 'Delete\nResource',
    #                         'create': 'Create\nResource',
    #                         'download': 'Download\nResource',
    #                         'app_launch': 'App\nLaunch'
    #                         })

    # modify the index to string type - for table printing
    df.index = [item.strftime('%m-%d-%Y') for item in df.index]

    # reverse the rows so that the table will be created in descending
    # chronological order
    df = df[::-1]
   
#    import pdb; pdb.set_trace()

    # create plot object
    # x = ds.index
    # y = ds.values.tolist()
    return plot.PlotObject(None, None,
                           label=kwargs.get('label', None),
                           dataframe=df)


def total(input_directory='.',
          start_time=datetime(2000, 1, 1),
          end_time=datetime(2030, 1, 1),
          label='',
          aggregation='1M',
          linestyle='-',
          color='k',
          **kwargs):

    print('--> calculating total distinct organizations')

    # load the data based on working directory and subset it if necessary
    df = load_data(input_directory)
    df = subset_by_date(df, start_time, end_time)

    # drop duplicates (except the first occurrence)
    df = df.drop_duplicates(subset='usr_organization', keep='first')

    # group and cumsum
    df = df.sort_index()
    ds = df.groupby(pandas.Grouper(freq=aggregation)) \
           .usr_organization.nunique().cumsum()

    # create plot object
    x = ds.index
    y = ds.values.tolist()
    return plot.PlotObject(x,
                           y,
                           label=label,
                           linestyle=linestyle,
                           color=color,
                           )


def us_universities(input_directory='.',
                    start_time=datetime(2000, 1, 1),
                    end_time=datetime(2030, 1, 1),
                    label='',
                    aggregation='1M',
                    linestyle='-',
                    color='k',
                    **kwargs):

    print('--> calculating distinct US universities')

    # load the data based on working directory and subset it if necessary
    df = load_data(input_directory)
    df = subset_by_date(df, start_time, end_time)

    # drop duplicates (except the first occurrence)
    df = df.drop_duplicates(subset='usr_organization', keep='first')

    # load university data
    uni = pandas.read_csv('dat/university-data.csv')
    uni_us = list(uni[uni.country == 'us'].university)

    # subset all organizations to just the approved list of US orgs
    df_us = df[df.usr_organization.isin(uni_us)]

    # group, cumulative sum, and create plot object
    df_us = df_us.sort_index()
    ds_us = df_us.groupby(pandas.Grouper(freq=aggregation)) \
                 .usr_organization.nunique().cumsum()
    x = ds_us.index
    y = ds_us.values.tolist()

    return plot.PlotObject(x,
                           y,
                           label=label,
                           linestyle=linestyle,
                           color=color)


def international_universities(input_directory='.',
                               start_time=datetime(2000, 1, 1),
                               end_time=datetime(2030, 1, 1),
                               label='',
                               aggregation='1M',
                               linestyle='-',
                               color='k',
                               **kwargs):


    print('--> calculating distinct international universities')

    # load the data based on working directory and subset it if necessary
    df = load_data(input_directory)
    df = subset_by_date(df, start_time, end_time)

    # drop duplicates (except the first occurrence)
    df = df.drop_duplicates(subset='usr_organization', keep='first')

    # load university data
    uni = pandas.read_csv('dat/university-data.csv')
    uni_int = list(uni[uni.country != 'us'].university)

    # subset all organizations to just the approved list of international orgs
    df_int = df[df.usr_organization.isin(uni_int)]

    # group, cumulative sum, and create plot object
    df_int = df_int.sort_index()
    ds_int = df_int.groupby(pandas.Grouper(freq=aggregation)) \
                   .usr_organization.nunique().cumsum()
    x = ds_int.index
    y = ds_int.values.tolist()

    return plot.PlotObject(x,
                           y,
                           label=label,
                           linestyle=linestyle,
                           color=color)


def cuahsi_members(input_directory='.',
                   start_time=datetime(2000, 1, 1),
                   end_time=datetime(2030, 1, 1),
                   label='',
                   aggregation='1M',
                   linestyle='-',
                   color='k',
                   **kwargs):

    print('--> calculating CUAHSI members')

    # load the data based on working directory and subset it if necessary
    df = load_data(input_directory)
    df = subset_by_date(df, start_time, end_time)

    # drop duplicates (except the first occurrence)
    df = df.drop_duplicates(subset='usr_organization', keep='first')

    # load cuahsi member data
    mem = pandas.read_csv('dat/cuahsi-members.csv')
    mems = list(mem.name)

    # subset all organizations to just the approved list of CUAHSI orgs
    df_mem = df[df.usr_organization.isin(mems)]

    # group, cumulative sum, and create plot object
    df_mem = df_mem.sort_index()
    ds_mem = df_mem.groupby(pandas.Grouper(freq=aggregation)) \
                   .usr_organization.nunique().cumsum()
    x = ds_mem.index
    y = ds_mem.values.tolist()

    return plot.PlotObject(x,
                           y,
                           label=label,
                           linestyle=linestyle,
                           color=color)

def org_type(input_directory='.',
             start_time=datetime(2000, 1, 1),
             end_time=datetime(2030, 1, 1),
             aggregation='1D',
             linestyle='-',
             orgtypes=[],
             users_threshold=50,
             **kwargs):

    # load the data based on working directory
    df = load_data(input_directory)
    df = subset_by_date(df, start_time, end_time)

    # define Org types
    # org_vocab = ['Unspecified',
    #             'Post-Doctoral Fellow',
    #             'Commercial/Professional',
    #             'University Faculty',
    #             'Government Official',
    #             'University Graduate Student',
    #             'Professional',
    #             'University Professional or Research Staff',
    #             'Local Government',
    #             'University Undergraduate Student',
    #             'School Student Kindergarten to 12th Grade',
    #             'School Teacher Kindergarten to 12th Grade',
    #             'Other']

    # # clean the data
    # df.loc[~df.usr_organization.isin(org_vocab), 'usr_organization'] = 'Other'
    df = df.loc[df.usr_organization != 'None']

    # loop through each of the user types
    plots = []

    # plot only the provided org types
    if len(orgtypes) == 0:
        # select all unique user types
        orgtypes = df.usr_organization.unique()

    colors = iter(cm.jet(numpy.linspace(0, 1, len(orgtypes))))
    for otype in orgtypes:
        if otype == 'None':
            continue

        # group by org type
        du = df.loc[df.usr_organization == otype]
        if du.shape[0] < users_threshold:
            next(colors)
            continue

        # remove null values
        # du = du.dropna()

        # group by date frequency
        ds = du.groupby(pandas.Grouper(freq=aggregation)).count().usr_organization.cumsum()
        x = ds.index
        y = ds.values
        c = next(colors)

        # create plot object
        plots.append(plot.PlotObject(x, y, label=otype, color=c, linestyle=linestyle))

    return plots