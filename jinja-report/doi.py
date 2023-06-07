#!/usr/bin/env python3

import os
import pickle
#import signal
import requests
#import argparse
import pandas as pd
import numpy
from lxml import etree
import hs_restclient as hsapi
from datetime import datetime
#import matplotlib.pyplot as plt
#import matplotlib.dates as mdates
from urllib3.exceptions import InsecureRequestWarning
from pandas.plotting import register_matplotlib_converters
from organizations import load_data as org_load_data
from matplotlib.pyplot import cm
import users


import plot
import creds
import utilities

register_matplotlib_converters()
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)


def load_data(working_dir, pickle_file='doi.pkl'):

    # return existing doi data
    path = os.path.join(working_dir, pickle_file)
    if os.path.exists(path):
        return pd.read_pickle(path)

    # doi data was not found, download it.
    print('--> collecting doi\'s from HydroShare')
    hs = hsapi.HydroShare(hostname='www.hydroshare.org',
                          prompt_auth=False,
                          use_https=False,
                          verify=False)
    resources = hs.resources(published=True)
    dat = []
    for resource in resources:
        meta = hs.getScienceMetadata(resource['resource_id'])
        dates = {}
        for dt in meta['dates']:
            dates[dt['type']] = dt['start_date']
        if 'published' not in dates:
            continue
        
        dat.append(dict(resource_id=resource['resource_id'],
                        owner_id=meta['creators'][0]['hydroshare_user_id'],
                        created_dt=dates['created'],
                        last_modified_dt=dates['modified'],
                        published_dt=dates['published'],
                        doi=resource['doi']))

    df = pd.DataFrame(dat)

    # convert columns to datetime objects
    df['created_dt'] = pd.to_datetime(df['created_dt'])
    df['last_modified_dt'] = pd.to_datetime(df['last_modified_dt'])
    df['Date Published'] = pd.to_datetime(df['published_dt'])

    with open(os.path.join(working_dir, 'doi.pkl'), 'wb') as f:
        pickle.dump(df, f)

    return df


def published_resources(input_directory='.',
                        start_time=datetime(2000, 1, 1),
                        end_time=datetime(2030, 1, 1),
                        aggregation='1M',
                        label='Monthly published resources',
                        color='b',
                        **kwargs):
    print(f'--> calculating published resources per {aggregation}')
    df = load_data(input_directory)
    df = utilities.subset_by_date(df,
                                  start_time,
                                  end_time,
                                  date_column='Date Published')
    df.set_index('Date Published', inplace=True)

    # group and sum
    df = df.sort_index()
    df = df.groupby(pd.Grouper(freq=aggregation)).count()['resource_id']

    # create plot object
    x = df.index
    y = df.values.tolist()

    return plot.PlotObject(x, y,
                           label=label,
                           color=color)

def publications_by_usr_type(input_directory='.',
                        start_time=datetime(2000, 1, 1),
                        end_time=datetime(2030, 1, 1),
                        aggregation='1M',
                        usertypes=[],
                        label='Monthly published resources',
                        color='b',
                        **kwargs):
    print(f'--> calculating published resources per {aggregation}')
    df = load_data(input_directory)
    df = utilities.subset_by_date(df,
                                  start_time,
                                  end_time,
                                  date_column='Date Published')
    
    # link owner_id to a user and get their institution
    org_df = org_load_data(input_directory)
    org_df['usr_id'] = pd.to_numeric(org_df['usr_id'],errors='coerce')
    df = df.merge(org_df, right_on='usr_id', left_on='owner_id', how='left')
    df.set_index('Date Published', inplace=True)
    # now we have usr_organization and usr_type

    # define HS user types
    ut_vocab = ['Post-Doctoral Fellow',
                'Commercial/Professional',
                'University Faculty',
                'Government Official',
                'University Graduate Student',
                'Professional',
                'University Professional or Research Staff',
                'Local Government',
                'University Undergraduate Student',
                'School Student Kindergarten to 12th Grade',
                'School Teacher Kindergarten to 12th Grade',
                'Other']

    # clean the data
    df.loc[~df.usr_type.isin(ut_vocab), 'usr_type'] = 'Other'
    
    # loop through each of the user types
    plots = []

    # plot only the provided user types
    if len(usertypes) == 0:
        # select all unique user types
        usertypes = df.usr_type.unique()

    colors = iter(cm.jet(numpy.linspace(0, 1, len(usertypes))))
    for utype in usertypes:

        # group by user type
        du = df.loc[df.usr_type == utype]

        # remove null values
        # du = du.dropna()

        # group by date frequency
        ds = du.groupby(pd.Grouper(freq=aggregation)).count().usr_type.cumsum()
        x = ds.index
        y = ds.values
        c = next(colors)

        # create plot object
        plots.append(plot.PlotObject(x, y, label=utype, color=c, linestyle='-'))

    return plots

def publications_by_org_type(input_directory='.',
                        start_time=datetime(2000, 1, 1),
                        end_time=datetime(2030, 1, 1),
                        aggregation='1M',
                        orgtypes=[],
                        label='Monthly published resources',
                        **kwargs):
    print(f'--> calculating published resources per {aggregation}')
    df = load_data(input_directory)
    df = utilities.subset_by_date(df,
                                  start_time,
                                  end_time,
                                  date_column='Date Published')
    
    # link owner_id to a user and get their institution
    org_df = org_load_data(input_directory)
    org_df['usr_id'] = pd.to_numeric(org_df['usr_id'],errors='coerce')
    df = df.merge(org_df, right_on='usr_id', left_on='owner_id', how='left')
    df.set_index('Date Published', inplace=True)
    # now we have usr_organization and usr_type

    # clean the data
    df = df.loc[df.usr_organization != 'None']
    
    # loop through each of the user types
    plots = []

    # plot only the provided org types
    if len(orgtypes) == 0:
        # select all unique user types
        orgtypes = df.usr_organization.unique()

    colors = iter(cm.jet(numpy.linspace(0, 1, len(orgtypes))))
    for otype in orgtypes:

        # group by org type
        du = df.loc[df.usr_organization == otype]

        # remove null values
        # du = du.dropna()

        # group by date frequency
        ds = du.groupby(pd.Grouper(freq=aggregation)).count().usr_organization.cumsum()
        x = ds.index
        y = ds.values
        c = next(colors)

        # create plot object
        plots.append(plot.PlotObject(x, y, label=otype, color=c, linestyle='-'))

    return plots

def orgpie(input_directory='.',
        start_time=datetime(2000, 1, 1),
        end_time=datetime(2030, 1, 1),
        exclude=['None'],
        publication_threshold=1,
        **kwargs):

    print('--> building doi org types pie-chart')

    df = _join_orgs_with_dois(input_directory,
        start_time, end_time,
        exclude,
        publication_threshold,
        **kwargs)

    return plot.PlotObject(None,
                           df.percent,
                           dataframe=df,
                           label='percent')

def orgtable(input_directory='.',
        start_time=datetime(2000, 1, 1),
        end_time=datetime(2030, 1, 1),
        exclude=['None'],
        publication_threshold=1,
        **kwargs):

    print('--> building doi org types pie-chart')

    df = _join_orgs_with_dois(input_directory,
        start_time, end_time,
        exclude,
        publication_threshold,
        **kwargs)
    
    df.reset_index(inplace=True)
    df = df.rename(columns = {'type':'Organizations', 'score':'Score', 'percent':'Percent'})

    return plot.PlotObject(None, None,
                           label=kwargs.get('label', 'Percent'),
                           dataframe=df)

def _join_orgs_with_dois(input_directory='.',
        start_time=datetime(2000, 1, 1),
        end_time=datetime(2030, 1, 1),
        exclude=['None'],
        publication_threshold=1,
        **kwargs):
    df = load_data(input_directory)
    df = utilities.subset_by_date(df,
                                  start_time,
                                  end_time,
                                  date_column='Date Published')
    
    # link owner_id to a user and get their institution
    org_df = org_load_data(input_directory)
    org_df['usr_id'] = pd.to_numeric(org_df['usr_id'],errors='coerce')
    df = df.merge(org_df, right_on='usr_id', left_on='owner_id', how='left')
    df.set_index('Date Published', inplace=True, drop=True)
    # now we have usr_organization and usr_type

    # clean the data
    df = df.loc[df.usr_organization != 'None']
    # df.dropna(inplace=True, subset=['usr_organization'])
    df = df.filter(items=['usr_organization'])

    # count number of users for each type
    orgtypes = df.usr_organization.unique()
    for o in orgtypes:
        df[o] = numpy.where(df['usr_organization'] == o, 1, 0)
    df['Other'] = numpy.where(~df['usr_organization'].isin(orgtypes), 1, 0)

    # remove 'usr_organization' b/c it's no longer needed
    # df = df.drop('usr_organization', axis=1)

    # remove specified columns so they won't be plotted
    for drp in exclude:
        try:
            print('--> not reporting %s: %s users'
                  % (drp, df[drp].sum()))
            df.drop(drp, inplace=True, axis=1)
        except Exception as e:
            print(f'Error dropping from doi orgs pie df: {e}')

    # calculate total and percentages for each user type
    ds = df.sum()

    df = pd.DataFrame({'type': ds.index, 'score': ds.values})
    df = df.set_index('type')
    df['percent'] = round(df['score']/df['score'].sum()*100, 2)

    print('--> total number of users reporting: %d' % df.score.sum())
    others_pct = 0
    others_score = 0
    for u in orgtypes:
        if u not in exclude:
            pct = df.loc[u].percent
            score = df.loc[u].score
            if pct < publication_threshold:
                others_pct += pct
                others_score += score
                print(f"Dropping Org: {u} because it has less than {publication_threshold}")
                try:
                    df = df.drop(u)
                except ValueError:
                    print(f"Error dropping {u}")
                    df = df.rename({u: '%s (%2.2f%%)' % (u, pct)})
            else:   
                df = df.rename({u: '%s (%2.2f%%)' % (u, pct)})
    print(f"Percent of values aggregated into 'other' = {others_pct}")
    other_index = 'Other (%2.2f%%)' % (others_pct)
    df.loc['Other', 'percent'] = others_pct
    df.loc['Other', 'score'] = others_score
    df = df.rename({'Other': other_index})
    df.sort_values('score', inplace=True, ascending=False)
    return df


def citations(input_directory='.',
              start_time=datetime(2000, 1, 1),
              end_time=datetime(2030, 1, 1),
              label='Citations of published resources',
              color='b',
              **kwargs):

    print(f'--> calculating published resource citation count')

    if not os.path.exists(os.path.join(input_directory, 'doi-citations.pkl')):
        df = load_data(input_directory)
        df = utilities.subset_by_date(df,
                                      start_time,
                                      end_time,
                                      date_column='Date Published')
        df.set_index('Date Published', inplace=True)
        df['citations'] = 0
        df['citing_dois'] = ''

        # collect citations for each doi
        for idx, row in df.iterrows():
            doi = row['doi']
            url = ('https://doi.crossref.org/servlet/getForwardLinks?'
                   f'usr={creds.username}&pwd={creds.password}&'
                   f'doi={doi}')
            try:
                res = requests.get(url)
                root = etree.fromstring(res.text.encode())
                citations = root.findall('.//body/forward_link',
                                        namespaces=root.nsmap)
                dois = []
                for citation in citations:
                    doi = citation.find('.//doi', namespaces=root.nsmap).text
                    dois.append(doi)

                df.at[idx, 'citations'] = len(dois)
                df.at[idx, 'citing_dois'] = ','.join(dois)
            except ConnectionError as e:
                print(f'Warning: issue with DOI connection: {e}')
                continue

        df.to_pickle(os.path.join(input_directory, 'doi-citations.pkl'))
    else:
        df = pd.read_pickle(os.path.join(input_directory,
                                         'doi-citations.pkl'))

    # generate plot
    x = df.index
    y = df.citations.tolist()

    return plot.PlotObject(x, y,
                           dataframe=df,
                           label=label,
                           color=color)
