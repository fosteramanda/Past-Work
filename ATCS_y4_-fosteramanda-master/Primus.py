#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This program explores drugs and public drunkness/disorderly conduct in the Tenderloin and whether or not there are more incidents
of both in this neighborhood. I found by analyizing total incidents and incident per capita that ultimately the Tenderloin had 
the most drug incidents of each of the neighboorhoods but not the most incidents of disorderly conduct. 
@author: amandafoster
"""
__version__ = '0.1'
__author__ = 'Amanda Foster'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import the 3 csv data files
sfcrime = pd.read_csv("SFCrime__2018_to_Present.csv")
sf_neighborhood = pd.read_csv("SF_neighbordhoods_2016.csv", index_col='Neighborhood')
sfdistricts = pd.read_csv("SF_Police_Districts.csv", index_col='PdDistrict')

""" This method graphs the total drug incidents by SF police district
"""
def graph_drug_incidents():
    drugs = sfcrime[ sfcrime['Incident Category'].isin(['Drug Violation','Drug Offense'])] ## erased any rows where incident isn't drug violation/offense
    drugs_district = drugs.groupby('Police District').aggregate('count')
    plotme = drugs_district.sort_values('Incident Category',ascending=False)
    plotme['Color'] = 'b'
    plotme.loc['Tenderloin','Color'] = 'y' #makes the Tenderloin a different color
    plotme.plot(y='Incident Category',kind='bar', color = plotme['Color'])
    plt.title("Drug Incidents by Police District in SF for 2018")
    plt.ylabel("Total Number of Incidents")
    plt.show()

""" This method graphs the total disorderly conduct incidents by SF police district
"""
def graph_disorderly_conduct_incidents(): 
    conduct = sfcrime[ sfcrime['Incident Category'] == 'Disorderly Conduct'] ## erased any rows where incident isn't disorderly conduct
    conduct_district = conduct.groupby('Police District').aggregate('count')
    plotme = conduct_district.sort_values('Incident Category',ascending=False)
    plotme['Color'] = 'b'
    plotme.loc['Tenderloin','Color'] = 'y' #makes the Tenderloin a different color
    plotme.plot(y='Incident Category',kind='bar', color = plotme['Color'])
    plt.ylabel("Total Number of Incidents")
    plt.title("Disorderly Conduct Incidents by Police District in SF for 2018")
    plt.show()

""" This method graphs the total drug incidents per Capita in SF neighboorhoods
"""
def graph_drugs_per_capita():
    crime_distcat = pd.crosstab(index=sfcrime["Analysis Neighborhood"],
                            columns=sfcrime["Incident Category"])
    crime_neigh = pd.concat([crime_distcat, sf_neighborhood], axis=1)
    crime_neigh['total drugs'] = crime_neigh['Drug Violation'] + crime_neigh['Drug Offense']
    crime_neigh['per_capita']= crime_neigh['total drugs']/crime_neigh['Population']
    plotme = crime_neigh.sort_values('per_capita',ascending=False)
    plotme['Color'] = 'b'
    plotme.loc['Tenderloin','Color'] = 'y'
    plotme.plot(y='per_capita',kind='bar', color = plotme['Color'])
    plt.title("Drug Incidents Per Capita in each SF Neighborhood for 2018")
    plt.xlabel("SF Neighborhoods")
    plt.ylabel("Drug Incidents Per Capita in SF for 2018")
    plt.show()

""" This method graphs the total disorderly conduct incidents per Capita in SF neighboorhoods
"""
def graph_disorderly_conduct_per_capita():
    crime_neighcat = pd.crosstab(index=sfcrime["Analysis Neighborhood"],
                            columns=sfcrime["Incident Category"])
    crime_neigh = pd.concat([crime_neighcat, sf_neighborhood], axis=1) 
    crime_neigh['per_capita']= crime_neigh['Disorderly Conduct']/crime_neigh['Population']
    plotme = crime_neigh.sort_values('per_capita',ascending=False)
    plotme['Color'] = 'b'
    plotme.loc['Tenderloin','Color'] = 'y'
    plotme.plot(y='per_capita',kind='bar', color = plotme['Color'])
    plt.title("Disorderly Conduct Incidents Per Capita in each SF Neighborhood for 2018")
    plt.xlabel("SF Neighborhoods")
    plt.ylabel("Disorderly Conduct Incidents Per Capita")
    plt.show()

""" This method graphs total crime incidents vs density for each police district in SF
"""
def graph_density_and_crime():
    crime_distcat = pd.crosstab(index=sfcrime["Police District"],
                            columns=sfcrime["Incident Category"])
    crime_distcat['Total'] = crime_distcat.apply(sum, axis=1)
    sfcrime_districts = pd.concat( [crime_distcat, sfdistricts], axis=1, sort=False)
    sfcrime_districts['Density']=sfcrime_districts['Population']/sfcrime_districts['Land Mass'] 
    plt.scatter(x = sfcrime_districts['Density'], y = sfcrime_districts['Total'], c= sfcrime_districts['Neighborhoods'].apply(tenderloin_to_color))
    plt.title("Total Crime Incidents vs Police District Density in 2018")
    plt.xlabel("Density of Police District")
    plt.ylabel("Total Crime Incidents")
    plt.show()

""" This method graphs total drug incidents vs density for each police district in SF
"""    
def graph_density_drugs():
    crime_distcat = pd.crosstab(index=sfcrime["Police District"],
                            columns=sfcrime["Incident Category"])
    crime_neigh = pd.concat([crime_distcat, sfdistricts], axis=1)
    crime_neigh['total drugs'] = crime_neigh['Drug Violation'] + crime_neigh['Drug Offense']
    crime_neigh['density']= crime_neigh['Population']/crime_neigh['Land Mass']
    plt.scatter(x = crime_neigh['density'], y = crime_neigh['total drugs'], c= crime_neigh['Neighborhoods'].apply(tenderloin_to_color))
    plt.title("Number of Drug Incidents vs Police District Density in 2018")
    plt.xlabel("Density of Police District")
    plt.ylabel("Total Drug Incidents")
    plt.show()

""" This method graphs total disorderly conduct incidents vs density for each police district in SF
"""
def graph_density_disorderly_conduct():
    crime_distcat = pd.crosstab(index=sfcrime["Police District"],
                            columns=sfcrime["Incident Category"])
    crime_neigh = pd.concat([crime_distcat, sfdistricts], axis=1)
    crime_neigh['total disorderly conduct'] = crime_neigh['Disorderly Conduct'] 
    crime_neigh['density']= crime_neigh['Population']/crime_neigh['Land Mass']
    plt.scatter(x = crime_neigh['density'], y = crime_neigh['total disorderly conduct'], c= crime_neigh['Neighborhoods'].apply(tenderloin_to_color))
    plt.title("Number of Disorderly Conduct Incidents vs Police District Density in 2018")
    plt.xlabel("Density of Police District")
    plt.ylabel("Total Disorderly Conduct Incidents")
    plt.show()
    

""" This method assigns the Tenderloin neighborhood the color red and others blue
"""
def tenderloin_to_color (res):
    res_colors = {  'Tenderloin' : 'red' } 
    if res in res_colors:
        return res_colors[res]
    else:
        return 'blue'

""" Main Method 
"""
if __name__ == "__main__":
    
    #draws all presentation graphs
    graph_drug_incidents()
    graph_disorderly_conduct_incidents()
    graph_drugs_per_capita()
    graph_disorderly_conduct_per_capita()
    graph_density_drugs()
    graph_density_disorderly_conduct()
    graph_density_and_crime()



