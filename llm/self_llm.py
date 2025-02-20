#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   self_llm.py
@Time    :   2023/10/16 18:48:08
@Author  :   Logan Zou 
@Version :   1.0
@Contact :   loganzou0421@163.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   在 LangChain LLM 基础上封装的项目类，统一了 GPT、文心、讯飞、智谱多种 API 调用
'''

from langchain.llms.base import LLM
from typing import Dict, Any, Mapping
from pydantic import Field

class Self_LLM(LLM):
    '''
    Custom LLM adapter class for standardizing API calls across different LLMs.
    Inherits from langchain.llms.base.LLM to maintain LangChain compatibility.
    '''

    # Native API endpoint
    url : str =  None
    # Default model
    model_name: str = "gpt-3.5-turbo"
    # Request timeout limit
    request_timeout: float = None
    # Temperature coefficient
    temperature: float = 0.1
    # API_Key
    api_key: str = None
    # Optional parameters
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)


    @property
    def _default_params(self) -> Dict[str, Any]:
        '''
        Defines a method that returns the default parameters
        '''
        normal_params = {
            "temperature": self.temperature,
            "request_timeout": self.request_timeout,
            }
        return {**normal_params}
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        '''
        Get the identifying parameters
        '''
        return {**{"model_name": self.model_name}, **self._default_params}

    # Example:
    # llm = Self_LLM(model_name="gpt-4", temperature=0.7)
    # print(llm._identifying_params)
    # Output: {'model_name': 'gpt-4', 'temperature': 0.7, 'request_timeout': None}
