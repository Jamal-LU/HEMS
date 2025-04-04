# -*- coding: utf-8 -*-


import random
import numpy as np
import pandas as pd
import gymnasium
from operator import truediv
import os
import pickle
import time
import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd
from torch.nn.modules import loss
import pandas as pd
from copy import deepcopy
from gymnasium.spaces import Box

battery_parameters={
'capacity':10,
'max_charge':3*.25,
'max_discharge':3*.25,
'max_Reactive':3*.25 ,
'efficiency':0.95,

'max_soc':1.0,
'min_soc':0.2,
'initial_capacity':np.random.normal(0.4, 0.05),
'conveter_capacity':3*.25}



EV_parameters={
'EV_capacity':16,
'EV_max_charge':3.3*.25,
'EV_max_discharge':3.3*.25,
'EV_Reactive':3.3*.25,
'EV_efficiency':.95,
'EV_max_soc':1.0,
'EV_min_soc':0.2,
'EV_initial_capacity':np.random.normal(0.4, 0.05),
'EV_leaving_capacity': 0.7*16,
'conveter_capacity':3.3*.25,
'EV_leaving_Time': random.randint(50, 60)}


class Constant:
	MONTHS_LEN = [26, 26, 26, 26, 26, 26, 26, 26, 26]
	MAX_STEP_HOURS = 96 * 30

class EV_avaliablity:
  def create_list_with_zeros(total_length=96, zero_length=35):
    # Create a list of 61 ones
    numbers = [1] * total_length
    # Randomly choose a start index for the zero sequence
    start_index = random.randint(0, total_length - zero_length)
    # Replace part of the list with 35 zeros
    for i in range(start_index, start_index + zero_length):
        numbers[i] = 0
    return numbers


class Price_electritiy:
  Price_time=[      0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16,
                    0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.11, 0.11, 0.11, 0.11, 0.11,
                    0.11, 0.11, 0.11, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
                    0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
                    0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.16, 0.16,
                    0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16,
                    0.16, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11,
                    0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11,
                    0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11]


Critical_load_parameters={
'fridge_power'       :0.2,# KW
'TV_power'           :0.1, # KW
'TV_intial_start'    :40, # 96 timesteps
'TV_end_start'       :44, # 96 timesteps
'TV_working_steps'   :16, # 96 timesteps
'Light_power'        :0.2, # KW
'Light_intial_start' :36, # 96 timesteps
'Light_end_start'    :40, # 96 timesteps
'Light_working_steps':20 # 96 timesteps
 }

Shiftable_load_parameters={
'Dishwasher_power'             :0.6,# KW
'Dishwasher_working_time'      :2,#  2 timesteps (half hour)
'Dishwasher_intial_start'      :0, # 96 timesteps
'Dishwasher_end_start'         :5, # 96 timesteps
'Dishwasher_working_steps'     :32, # 96 timesteps

'Washer_power'                 :0.38,# KW
'Washer_working_time'          :4,#  ( hour)
'Washer_intial_start'          :0, # 96 timesteps
'Washer_end_start'             :5, # 96 timesteps
'Washer_working_steps'         :16, # 96 timesteps

'Drayer_power'                 :1.2,# KW
'Drayer_working_time'          :4,# 1 hours
'Drayer_intial_start'          :20, # 96 timesteps
'Drayer_end_start'             :24, # 96 timesteps
'Drayer_working_steps'         :20 # 96 timesteps
 }

class DataManager():
    def __init__(self) -> None:
        self.PV_Generation=[]
        self.Prices=[]
        self.Electricity_Consumption=[]
        self.Weather=[]
    def add_pv_element(self,element):
      self.PV_Generation.append(element)
    def add_wather_element(self,element):
      self.Weather.append(element)
    def add_price_element(self,element):
      self.Prices.append(element)
    def add_electricity_element(self,element):
      self.Electricity_Consumption.append(element)

    # get current time data based on given month day, and day_time
    def get_pv_data(self,month,day,day_time):
      return self.PV_Generation[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*96+day_time]
    def get_weather_data(self,month,day,day_time):
      return self.Weather[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*96+day_time]
    def get_price_data(self,month,day,day_time):
      return self.Prices[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*96+day_time]
    def get_electricity_cons_data(self,month,day,day_time):
      return self.Electricity_Consumption[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*96+day_time]
    # get series data for one episode
    def get_series_pv_data(self,month,day):
      return self.PV_Generation[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*96:(sum(Constant.MONTHS_LEN[:month-1])+day-1)*96+96]
    def get_series_weather_data(self,month,day):
      return self.Weather[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*96:(sum(Constant.MONTHS_LEN[:month-1])+day-1)*96+96]
    def get_series_price_data(self,month,day):
      return self.Prices[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*96:(sum(Constant.MONTHS_LEN[:month-1])+day-1)*96+96]
    def get_series_electricity_cons_data(self,month,day):
      return self.Electricity_Consumption[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*96:(sum(Constant.MONTHS_LEN[:month-1])+day-1)*96+96]

class Battery():
    def __init__(self,parameters):
        self.capacity=parameters['capacity']
        self.max_soc=parameters['max_soc']
        self.initial_capacity=parameters['initial_capacity']
        self.min_soc=parameters['min_soc']
        self.max_charge=parameters['max_charge']
        self.max_discharge=parameters['max_discharge']
        self.efficiency=parameters['efficiency']
        self.battery_compared=None
    def step(self,action_battery):
        energy=action_battery*self.max_charge
        updated_capacity=max(self.min_soc,min(self.max_soc,(self.current_capacity*self.capacity+energy)/self.capacity))
        self.energy_change=(updated_capacity-self.current_capacity)*self.capacity
        self.current_capacity=updated_capacity
    def SOC(self):
        return self.current_capacity
    def reset(self):
        self.current_capacity= np.random.uniform(0.1,1.0)

class EV():
    def __init__(self,parameters):
        self.EV_capacity=parameters['EV_capacity']
        self.EV_max_soc=parameters['EV_max_soc']
        self.EV_initial_capacity=parameters['EV_initial_capacity']
        self.EV_min_soc=parameters['EV_min_soc']# 0.2
        self.EV_max_charge=parameters['EV_max_charge']# nax charge ability
        self.EV_max_discharge=parameters['EV_max_discharge']
        self.EV_efficiency=parameters['EV_efficiency']
        self.EV_leaving_capacity=parameters['EV_leaving_capacity']
        self.time_step_EV=0
        self.EV_compared=None
    def step(self,action_EV):
        EV_energy=action_EV*self.EV_max_charge
        EV_updated_capacity=max(self.EV_min_soc,min(self.EV_max_soc,(self.EV_current_capacity*self.EV_capacity+EV_energy)/self.EV_capacity))
        self.EV_energy_change=(EV_updated_capacity-self.EV_current_capacity)*self.EV_capacity# if charge, positive, if discharge, negative
        if self.time_step_EV == np.random.uniform(50,55):
          EV_updated_capacity = np.random.uniform(0.2,0.4)

        self.EV_current_capacity=EV_updated_capacity #update capacity to current codition
        self.time_step_EV +=1
    def EV_SOC(self):
        return self.EV_current_capacity
    def reset(self):
        self.EV_current_capacity= np.random.uniform(0.1,1.0)

class AC():
  '''simulate an AC  here'''
  def __init__(self,parameters):
      self.Room_temp_wanted_max=parameters['Room_temp_wanted_max']
      self.Room_temp_max=parameters['Room_temp_max']
      self.Room_temp_wanted_min=parameters['Room_temp_wanted_min']
      self.Room_temp_min=parameters['Room_temp_min']
      self.initial_Room_temp=parameters['initial_Room_temp']
      self.max_AC_power=parameters['max_AC_power']
      self.current_time_temp=None
      self.Tem_room_NOW=None

  def step(self,action_AC, outside_temp_):
      self.Force_AC=action_AC*self.max_AC_power
      self.Tem_room_next=self.Tem_room_NOW-(((self.Tem_room_NOW-outside_temp_-29.7*self.Force_AC))/4.455)
      if self.Tem_room_next > self.Room_temp_max:
        self.Tem_room_next=self.Room_temp_max
      elif self.Tem_room_next < self.Room_temp_min:
        self.Tem_room_next=self.Room_temp_min
      self.Tem_room_NOW=self.Tem_room_next

  def temp_value(self):
      return  self.Tem_room_NOW

  def reset(self):
     self.Tem_room_NOW=np.random.uniform(69,77)
     self.Force_AC=0


class Shiftable_load():
  '''simulate a  Shiftable_load here'''
  def __init__(self,parameters):
      self.Dishwasher_power=parameters['Dishwasher_power'] #0.6 KW
      self.Dishwasher_working_time=parameters['Dishwasher_working_time'] # 8 [2 hours working]
      self.Dishwasher_intial_start=parameters['Dishwasher_intial_start'] #0  [start]
      self.Dishwasher_end_start=parameters['Dishwasher_end_start'] # 5       [another start]
      self.Dishwasher_working_steps=parameters['Dishwasher_working_steps'] # 16 [time winodw]
      self.Dishwasher_Load_total=np.array([])
      self.Dishwasher_Load_ins=None
      self.Dishwasher_bn_sum=0
      self.BN_Dishwahser=None
      self.time_Dishwahser_state=None
      self.k_Dishwasher= self.Dishwasher_working_time

      self.Washer_power=parameters['Washer_power'] #0.38 KW
      self.Washer_working_time=parameters['Washer_working_time'] # 16   [4 hours working]
      self.Washer_intial_start=parameters['Washer_intial_start'] #0     [start]
      self.Washer_end_start=parameters['Washer_end_start'] # 5          [another start]
      self.Washer_working_steps=parameters['Washer_working_steps'] # 16 [time winodw]
      self.Washer_Load_total=np.array([])
      self.Washer_Load_ins=None
      self.Washer_bn_sum=0
      self.BN_Washer=None
      self.time_Washer_state=None
      self.k_Washer= self.Washer_working_time

      self.Drayer_power=parameters['Drayer_power'] #1.2 KW
      self.Drayer_working_time=parameters['Drayer_working_time'] # 8   [2 hours working]
      self.Drayer_intial_start=parameters['Drayer_intial_start'] #20   [start]
      self.Drayer_end_start=parameters['Drayer_end_start'] #24         [another start]
      self.Drayer_working_steps=parameters['Drayer_working_steps'] #20 [time winodw]
      self.Drayer_Load_total=np.array([])
      self.Drayer_Load_ins=None
      self.Drayer_bn_sum=0
      self.BN_Drayer=None
      self.time_Drayer_state=None
      self.k_Drayer= self.Drayer_working_time


      self.time_step_SL=0
  def step(self, Dishwasher_action,Washer_action, Drayer_action,  \
           Dishwasher_starting, Washer_starting, Drayer_starting):
      Dishwasher_ending= Dishwasher_starting+self.Dishwasher_working_steps
      if self.time_step_SL >= Dishwasher_starting and (Dishwasher_ending-self.time_step_SL >= self.k_Dishwasher):
        if Dishwasher_action==0:
          self.Dishwasher_Load_total=np.append(self.Dishwasher_Load_total,0)
        else:
          load_Dishwashers=np.array([self.Dishwasher_power]*self.Dishwasher_working_time)
          self.Dishwasher_Load_total=np.append(self.Dishwasher_Load_total,load_Dishwashers)
          self.k_Dishwasher=1000
      else:
        self.Dishwasher_Load_total=np.append(self.Dishwasher_Load_total,0)

      if self.time_step_SL >= Dishwasher_starting:
          self.time_Dishwahser_state=Dishwasher_ending-self.time_step_SL
          if self.time_Dishwahser_state<0:
            self.time_Dishwahser_state=0
      else:
        self.time_Dishwahser_state=0

      bn_Dishwahser=self.Dishwasher_Load_total[self.time_step_SL]
      if bn_Dishwahser>0:
        bn_Dishwahser=1
      else:
        bn_Dishwahser=0
      self.Dishwasher_bn_sum += bn_Dishwahser
      self.BN_Dishwahser= (self.Dishwasher_bn_sum/self.Dishwasher_working_time)
      self.Dishwasher_Load_ins=self.Dishwasher_Load_total[self.time_step_SL]
      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      Washer_ending= Washer_starting+self.Washer_working_steps
      if self.time_step_SL >= Washer_starting and (Washer_ending-self.time_step_SL >= self.k_Washer):
        if Washer_action==0:
          self.Washer_Load_total=np.append(self.Washer_Load_total,0)
        else:
          load_washers=np.array([self.Washer_power]*self.Washer_working_time)
          self.Washer_Load_total=np.append(self.Washer_Load_total,load_washers)
          self.k_Washer=1000
      else:
        self.Washer_Load_total=np.append(self.Washer_Load_total,0)

      if self.time_step_SL >= Washer_starting:
          self.time_Washer_state=Washer_ending-self.time_step_SL
          if self.time_Washer_state<0:
            self.time_Washer_state=0
      else:
            self.time_Washer_state=0

      bn_Washer=self.Washer_Load_total[self.time_step_SL]
      if bn_Washer>0:
        bn_Washer=1
      else:
        bn_Washer=0
      self.Washer_bn_sum += bn_Washer
      self.BN_Washer= (self.Washer_bn_sum/self.Washer_working_time)
      #print(f't= {self.time_step_SL}\t ,state(0,1) {bn_Washer}, \t ,sum state {self.Washer_bn_sum},\t ,D% {self.BN_Washer}')
      self.Washer_Load_ins=self.Washer_Load_total[self.time_step_SL]

     #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      Drayer_ending= Drayer_starting+self.Drayer_working_steps
      if self.time_step_SL >= Drayer_starting and (Drayer_ending-self.time_step_SL >= self.k_Drayer):
        if Drayer_action==0:
          self.Drayer_Load_total=np.append(self.Drayer_Load_total,0)
        else:
          load_Drayers=np.array([self.Drayer_power]*self.Drayer_working_time)
          self.Drayer_Load_total=np.append(self.Drayer_Load_total,load_Drayers)
          self.k_Drayer=1000
      else:
        self.Drayer_Load_total=np.append(self.Drayer_Load_total,0)

      if self.time_step_SL >= Drayer_starting:
          self.time_Drayer_state=Drayer_ending-self.time_step_SL
          if self.time_Drayer_state<0:
            self.time_Drayer_state=0
      else:
          self.time_Drayer_state=0

      bn_Drayer=self.Drayer_Load_total[self.time_step_SL]
      if bn_Drayer>0:
        bn_Drayer=1
      else:
        bn_Drayer=0
      self.Drayer_bn_sum += bn_Drayer
      self.BN_Drayer= (self.Drayer_bn_sum/self.Drayer_working_time)
      self.Drayer_Load_ins=self.Drayer_Load_total[self.time_step_SL]
      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      self.time_step_SL +=1

  def total_SL(self):
    return self.Dishwasher_Load_ins, self.Washer_Load_ins,self.Drayer_Load_ins, \
           self.BN_Dishwahser, self.BN_Washer, self.BN_Drayer, \
           self.time_Dishwahser_state, self.time_Washer_state, self.time_Drayer_state

  def reset(self):
      self.time_step_SL=0
      self.k_Dishwasher= self.Dishwasher_working_time
      self.k_Washer= self.Washer_working_time
      self.k_Drayer= self.Drayer_working_time
      self.Dishwasher_Load_ins, self.Washer_Load_ins,self.Drayer_Load_ins =0,0,0
      self.BN_Dishwahser, self.BN_Washer, self.BN_Drayer=0,0,0
      self.time_Dishwahser_state, self.time_Washer_state, self.time_Drayer_state=0,0,0

class Grid():
    def __init__(self):

        self.on=True
        if self.on:
            self.exchange_ability=1000
        else:
            self.exchange_ability=0
    def _get_cost(self,current_price,energy_exchange):
        return current_price*energy_exchange
    def retrive_past_price(self):
        result=[]
        if self.day<1:
            past_price=self.past_price#
        else:
            past_price=self.price[96*(self.day-1):24*self.day]
        for item in past_price[(self.time-96)::]:
            result.append(item)
        for item in self.price[96*self.day:(96*self.day+self.time)]:
            result.append(item)
        return result
class ESS_EV_PF_Env():
    def __init__(self,**kwargs):
        super(ESS_EV_PF_Env,self).__init__()
        #parameters
        self.data_manager=DataManager()
        self._load_year_data()
        self.episode_length=kwargs.get('episode_length',96)
        self.month=None
        self.day=None
        self.TRAIN=True
        self.month_compared=None
        self.day_compared=None
        self.battery_compared=None
        self.EV_compared=None
        self.current_time=None
        self.AC_parameters=kwargs.get('AC_parameters',AC_parameters)
        self.Critical_load_parameters=kwargs.get('Critical_load_parameters',Critical_load_parameters)
        self.Shiftable_load_parameters=kwargs.get('Shiftable_load_parameters',Shiftable_load_parameters)

        self.TV_intial_start=Critical_load_parameters['TV_intial_start']
        self.TV_end_start=Critical_load_parameters['TV_end_start']
        self.Light_intial_start=Critical_load_parameters['Light_intial_start']
        self.Light_end_start=Critical_load_parameters['Light_end_start']
        self.starting_TV_time=None
        self.starting_light_time=None


        self.Dishwasher_intial_start=Shiftable_load_parameters['Dishwasher_intial_start'] #40
        self.Dishwasher_end_start=Shiftable_load_parameters['Dishwasher_end_start'] #44
        self.starting_Dishwasher_time=None

        self.Washer_intial_start=Shiftable_load_parameters['Washer_intial_start'] #40
        self.Washer_end_start=Shiftable_load_parameters['Washer_end_start'] #44
        self.starting_washer_time=None

        self.Drayer_intial_start=Shiftable_load_parameters['Drayer_intial_start'] #40
        self.Drayer_end_start=Shiftable_load_parameters['Drayer_end_start'] #44
        self.starting_Drayer_time=None

        self.Room_temp_wanted_max=AC_parameters['Room_temp_wanted_max']
        self.Room_temp_wanted_min=AC_parameters['Room_temp_wanted_min']
        self.max_AC_power=AC_parameters['max_AC_power']
        self.modification_AC_power=AC_parameters['modification_AC_power']
        self.battery_parameters=kwargs.get('battery_parameters',battery_parameters)
        self.EV_parameters=kwargs.get('EV_parameters',EV_parameters)
        self.EV_avaliablity= EV_avaliablity.EV_avaliablity_time
        self.price_electritiy=Price_electritiy.Price_time
        self.penalty_coefficient=3000
        self.sell_coefficient=0.5
        self.comfortable=0
        self.penalty_tempture=20
        self.time_counting=0
        self.PF_value= None
        PF_base= 0.95

        self.grid=Grid()
        self.battery=Battery(self.battery_parameters)
        self.EV=EV(self.EV_parameters)
        self.AC=AC(self.AC_parameters)
        self.Critical_load=Critical_load(self.Critical_load_parameters)
        self.Shiftable_load=Shiftable_load(self.Shiftable_load_parameters)


        self.action_space=Box(low=-1,high=1,shape=(8,),dtype=np.float32)

        self.state_space=Box(low=-1,high=1,shape=(16,),dtype=np.float32)

    @property
    def netload(self):

        return self.demand-self.grid.wp_gen-self.grid.pv_gen

    def reset(self,):
        self.month=np.random.randint(0,9)
        self.starting_Drayer_time=np.random.randint(self.Drayer_intial_start,self.Drayer_end_start)
        self.starting_washer_time=np.random.randint(self.Washer_intial_start,self.Washer_end_start)
        self.starting_Dishwasher_time=np.random.randint(self.Dishwasher_intial_start,self.Dishwasher_end_start)
        if self.TRAIN:
            self.day=np.random.randint(1,26)
        else:
            self.day=26


        self.current_time=0
        self.battery.reset()
        self.EV.reset()
        self.Critical_load.reset()
        self.Shiftable_load.reset()
        return self._build_state()
    def _build_state(self):
        soc=self.battery.SOC()
        EV_soc=self.EV.EV_SOC()
        time_step=self.current_time
        EV_available= self.EV_avaliablity[time_step]
        electricity_demand=self.data_manager.get_electricity_cons_data(self.month,self.day,self.current_time)
        pv_generation=self.data_manager.get_pv_data(self.month,self.day,self.current_time)
        self.outside_temp=self.data_manager.get_weather_data(self.month,self.day,self.current_time)
        self.outside_next=self.data_manager.get_weather_data(self.month,self.day,self.current_time+1)
        price=self.price_electritiy[time_step]
        net_load=electricity_demand-pv_generation

        Critical_load_total=self.Critical_load.total_CL()
        Dishwasher_Load,Washer_Load,Drayer_Load, Dishwasher_percent,Washer_percent,Drayer_percent,\
        Dishwasher_timer,Washer_timer,Drayer_timer=self.Shiftable_load.total_SL()
        Total_load= Critical_load_total+Dishwasher_Load+Washer_Load+Drayer_Load
        time_step=self.current_time

        obs=np.concatenate((np.float32(time_step),np.float32(price),np.float32(soc),\
                             np.float32(self.PF_value), \
                             np.float32(self.outside_temp),np.float32(self.outside_next),np.float32(current_temp_room), \
                             (Dishwasher_percent),(Washer_percent),(Drayer_percent),\
                             (Dishwasher_timer), (Washer_timer),(Drayer_timer),
                            np.float32(EV_soc),np.float32(EV_available),np.float32(Total_load)),axis=None)
        return obs

    def step(self,action):

        current_obs=self._build_state()
        self.battery.step(action[0])
        self.EV.step(action[1]*current_obs[4])
        current_output=np.array((-self.battery.energy_change,-self.EV.EV_energy_change))
        self.current_output=current_output
        actual_production=sum(current_output)
        netload=current_obs[16]
        price=current_obs[1]
        self.AC.step(action[8],current_obs[6])
        unbalance=actual_production-netload
        reactive_load= (electricity_demand+Total_load) * tan_angle
        reward=0
        excess_penalty=0
        deficient_penalty=0
        sell_benefit=0
        buy_cost=0
        self.excess=0
        self.shedding=0
        confortablitiy_level=0
        angle = math.acos(PF_base)
        tan_angle = math.tan(angle)
        reactive_load_correction= reactive_load-abs(action[2])-abs(action[3])
        self.Critical_load.step(self.starting_TV_time,self.starting_light_time )
        self.Shiftable_load.step(action[5], action[6], action[7],\
                                 self.starting_Dishwasher_time,self.starting_washer_time,self.starting_Drayer_time)
        if unbalance>=0:
            if unbalance<=self.grid.exchange_ability:
                sell_benefit=self.grid._get_cost(price,unbalance)*self.sell_coefficient #sell money to grid is little [0.029,0.1]
            else:
                sell_benefit=self.grid._get_cost(price,self.grid.exchange_ability)*self.sell_coefficient

                self.excess=unbalance-self.grid.exchange_ability
                excess_penalty=self.excess*self.penalty_coefficient

        else:
            if abs(unbalance)<=self.grid.exchange_ability:
                buy_cost=self.grid._get_cost(price,abs(unbalance))
            else:
                buy_cost=self.grid._get_cost(price,self.grid.exchange_ability)

                #self.shedding=abs(unbalance)-self.grid.exchange_ability
                #deficient_penalty=self.shedding*self.penalty_coefficient

        if current_obs[0]== self.EV_parameters['EV_leaving_Time']:
          if current_obs[3] < self.EV_parameters['EV_leaving_capacity']:
            confortablitiy_level = (current_obs[3] - self.EV_parameters['EV_leaving_capacity'])* -self.penalty_coefficient

          else:
            confortablitiy_level = 0
        battery_cost=self.battery._get_cost(self.battery.energy_change)# we set it as 0 this time
        if reactive_load_correction > 0:
          if unbalance <= 0:
            unbalance=0.2
            complex_power= (unbalance**2)+(reactive_load_correction**2)
            PF= abs(unbalance)/((complex_power)**0.5)
            if PF < 0.9:
              if PF >= 0.3:
                PF_penalty= math.exp(1/PF)
              else:
                PF_penalty= math.exp(1/0.3)
          else:
            complex_power= (unbalance**2)+(reactive_load_correction**2)
            PF= abs(unbalance)/((complex_power)**0.5)
            if PF < 0.9:
              if PF >= 0.3:
                PF_penalty= math.exp(1/PF)
              else:
                PF_penalty= math.exp(1/0.3)
          self.PF_value= PF
        else:
          PF_penalty=0
          PF=1
          self.PF_value= PF

        if current_obs[0] == 94:
           #print('current_obs[3] , current_obs[4] , current_obs[5]',current_obs[3] , current_obs[4] , current_obs[5])
           if current_obs[5]==0:
             confortablitiy_level_SH= self.penalty_Shiftable_load
           if current_obs[6]==0:
             confortablitiy_level_SH= self.penalty_Shiftable_load
           if current_obs[7]==0:
             confortablitiy_level_SH= self.penalty_Shiftable_load


        if current_obs[8]> self.Room_temp_wanted_max:
          confortablitiy_Temp = math.exp(current_obs[8]-self.Room_temp_wanted_max)*self.penalty_tempture
        elif current_obs[8] < self.Room_temp_wanted_min:
          confortablitiy_Temp = math.exp(self.Room_temp_wanted_min-current_obs[8])*self.penalty_tempture
        reward-=(-sell_benefit-buy_cost+confortablitiy_level+self.PF_value+confortablitiy_level_SH+confortablitiy_Temp)/1e3
        #self.operation_cost=battery_cost+dg1_cost+dg2_cost+dg3_cost+buy_cost-sell_benefit+excess_penalty+deficient_penalty
        #self.unbalance=unbalance
        #self.real_unbalance=self.shedding+self.excess
        final_step_outputs=[self.battery.current_capacity, self.EV.EV_current_capacity]
        self.current_time+=1
        finish=(self.current_time==self.episode_length)
        self.confortablitiy_shiftable=confortablitiy_level
        self.confortablitiy_tempture=confortablitiy_Temp
        if finish:
            self.final_step_outputs=final_step_outputs
            self.current_time=0
            next_obs=self.reset()

        else:
            next_obs=self._build_state()
        return current_obs,next_obs,float(reward),finish
    def render(self, current_obs, next_obs, reward, finish):
        print('day={},hour={:2d}, state={}, next_state={}, reward={:.4f}, terminal={}\n'.format(self.day,self.current_time, current_obs, next_obs, reward, finish))

    def _load_year_data(self):
        pv_df=pd.read_csv('/content/PV.csv',sep=';')
        #hourly price data for a year
        price_df=pd.read_csv('/content/Prices.csv',sep=';')
        # Weather
        weather_df=pd.read_csv('/content/weather2.csv')
        # mins electricity consumption data for a year
        #electricity_df=pd.read_csv('/content/H4.csv',sep=';')
        pvs_data=pv_df['P_PV_'].apply(lambda x: x.replace(',','.')).to_numpy(dtype=float)
        pv_data=np.array([x for x in pvs_data for i in range(4)])
        prices=price_df['Price'].apply(lambda x:x.replace(',','.')).to_numpy(dtype=float)
        price=np.array([x for x in prices for i in range(4)])
        #electricitys=electricity_df['Power'].apply(lambda x:x.replace(',','.')).to_numpy(dtype=float)
        #electricity=np.array([x for x in electricitys for i in range(4)])
        weather_data=weather_df['Outdoor_temp'].to_numpy(dtype=float)
        weather_data_Fs=weather_data*1.8+32
        weather_data_F=np.array([x for x in weather_data_Fs for i in range(4)])
        # netload=electricity-pv_data

        for element in pv_data:
            self.data_manager.add_pv_element(element*200)

        for element in weather_data_F:
            self.data_manager.add_wather_element(element*1)
        for element in price:
            element/=10
            if element<=0.5:
                element=0.5
            self.data_manager.add_price_element(element)
