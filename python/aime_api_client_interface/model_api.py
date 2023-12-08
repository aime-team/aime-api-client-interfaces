import aiohttp
import base64
import asyncio
import requests
import time
import pickle
import pkg_resources


DEFAULT_PROGRESS_INTERVAL = 0.3


class ModelAPI():
    """
    An interface for interacting with the AIME ML API server.

    Args:
        api_server (str): The base URL of the API server.
        endpoint_name (str): The name of the API endpoint.
        session (aiohttp.ClientSession): Give existing session to ModelApi API to make login request in given session. Defaults to None.

    Attributes:
        api_server (str): The base URL of the API server.
        endpoint_name (str): The name of the specific API endpoint.
        client_session_auth_key (str): The authentication key for the client session, obtained from do_api_login.

    Examples:
    
        Synchronous with progress callback:

        .. highlight:: python
        .. code-block:: python
          
            from python_api_client_interface import ModelAPI
            
            def progress_callback(progress_info, progress_data):
                process_progress_info(progress_info)
                process_progress_data(progress_data)

            model_api = ModelAPI('https://api.aime.team', 'llama2_chat')
            model_api.do_api_login()
            result = model_api.do_api_request(params)
            result_2 = model_api.do_api_request(params)
            ...
            

        Asynchronous with asynchronous callbacks:

        .. highlight:: python
        .. code-block:: python

            import asyncio
            from python_api_client_interface import ModelAPI

            async def result_callback(result):
                await process_result(result)

            async def progress_callback(progress_info, progress_data):
                await process_progress_info(progress_info)
                await process_progress_data(progress_data)

            async def progress_error_callback(error_description):
                print(error_description)

            async def main():
                model_api = modelAPI('https://api.aime.team', 'llama2_chat')
                await model_api.do_api_login()
                result = await model_api.do_api_request(params, result_callback, progress_callback)
                result2 = await model_api.do_api_request(params, result_callback, progress_callback)
                ...
                await model_api.close_session()

            asynchio.run(main())


        Asynchronous with synchronous callbacks:

        .. highlight:: python
        .. code-block:: python

            import asyncio
            from python_api_client_interface import ModelAPI

            sync def result_callback(result):
                process_result(result)

            def progress_callback(progress_info, progress_data):
                process_progress_info(progress_info)
                process_progress_data(progress_data)

            def progress_error_callback(error_description):
                print(error_description)

            async def main():
                model_api = modelAPI('https://api.aime.team', 'llama2_chat')
                await model_api.do_api_login()
                result = await model_api.do_api_request(params, result_callback, progress_callback)
                result2 = await model_api.do_api_request(params, result_callback, progress_callback)
                ...
                await model_api.close_session()

            asynchio.run(main())

    """

    def __init__(self, api_server, endpoint_name, session=None, output_format='base64'):
        
        """
        Constructor

        Args:
            api_server (str): The base URL of the API server.
            endpoint_name (str): The name of the API endpoint.
            session (aiohttp.ClientSession): Give existing session to ModelApi API to make upcoming requests in given session. 
                Defaults to None.
            output_format (str) = Output format of objects like images in result dictionary of do_api_request() and do_api_request_async().
                Defaults to 'base64'.  
        """
        self.api_server = api_server
        self.endpoint_name = endpoint_name
        self.session = session
        self.client_session_auth_key = None
        self.output_format = output_format


    async def __aexit__(self):
        await self.session.close()


    async def do_api_login_async(
        self,
        result_callback=None,
        error_callback=None,
        session=None
        ):
        """
        Asynchronous client login to API server and obtain an authentication key.

        Args:
            result_callback (callable or coroutine): Callback function or coroutine with the obtained client_session_auth_key as argument. 
                Accepts synchronous functions and asynchrouns couroutines. Defaults to None.
            error_callback (callable or coroutine), Callback function or coroutine for catching errors obtaining client_session_auth_key. 
                Accepts synchronous functions and asynchrouns couroutines. Prevents ConnectionError. Defaults to None.
            session (aiohttp.ClientSession): Give existing session to ModelApi API to make login request in given session. Defaults to None.

        Raises:
            ConnectionError: If there is a connection issue with the API server and no error_callback given.

        Returns:
            str: Client session authentication key
        """
        self.setup_session(session)
        self.client_session_auth_key = await self.__fetch_auth_key_async(error_callback)
        return self.client_session_auth_key
        

    def do_api_login(self):
        """
        Client login to API server and obtain an authentication key.

        Raises:
            ConnectionError: If there is a connection issue with the API server.

        Returns:
            str: Client session authentication key
        """
        self.client_session_auth_key = self.__fetch_auth_key()
        return self.client_session_auth_key


    async def do_api_request_async(
        self,
        params,
        result_callback=None,
        progress_callback=None,
        request_error_callback=None,
        progress_error_callback=None,
        progress_interval=DEFAULT_PROGRESS_INTERVAL,
        progress_stream=False,
        session=None,
        output_format='base64'
        ):
        """
        Do an asynchronous API request with optional progress data via asynchronous or synchronous callbacks. 

        Args:
            params (dict): Dictionary with parameters for the the API request like 'prompt' or 'image'.
            result_callback (callable or coroutine, optional): Callback function or coroutine with argument result (dict) to handle the API request result.
                Accepts synchronous functions and asynchrouns couroutines. Defaults to None
            progress_callback (callable or coroutine, optional): Callback function or coroutine with arguments progress_info (dict) and 
                progress_data (dict) for tracking progress. Accepts synchronous functions and asynchrouns couroutines. Default is None.
            request_error_callback (callable or coroutine, optional): Callback function or coroutine with argument error_description (str) 
                for catching request errors. Accepts synchronous functions and asynchrouns couroutines.
                Prevents ConnectionError and PermissionError. Defaults to None.
            progress_error_callback (callable or coroutine, optional): Callback function or coroutine with argument error_description (str) for catching 
                progress errors with successful initial request. Accepts synchronous functions and asynchrouns couroutines.
                Prevents ConnectionErrors during Transmitting. Defaults to None.
            progress_interval (int, optional): Interval in seconds at which progress is checked. Default is {DEFAULT_PROGRESS_INTERVAL}.
            progress_stream (bool, optional): Not implemented yet
            session (aiohttp.ClientSession): Give existing session to ModelApi API to make login request in given session. Defaults to None.

        Raises:
            ConnectionError: Raised if client couldn't connect with API sserver and no request_error_callback is given. Also raised if client lost 
                connection during transmitting and no progress_error_callback is given.
            PermissionError: Raised if client is not logged in the API server and no error_callback given.
            NotImplementedError: If progress_stream != None.

        Returns:
            dict: Dictionary with job results

        Examples:

            Example job result dict in result callback argument or return value:

            .. highlight:: python
            .. code-block:: python

                result = {
                    'auth': '<name_of_worker>',
                    'compute_duration': 2.4,
                    'images': 'data:image/PNG;base64,...'
                    'job_id': 'JID3',
                    'seed': 1234413214,
                    'success': True,
                    'text': 'Test output...',
                    'total_duration': 2
                }

                
            Example progress_callback arguments progress_info and progress_data dictionaries:  

            .. highlight:: python
            .. code-block:: python
            

                progress_info = {
                    'job_id': 'JID3', 
                    'progress': 50, 
                    'queue_position': 0, 
                    'estimate': -1
                }
                progress_data = {
                        'info': '<infos from worker about progress', 
                        'images': 'data:image/PNG;base64,...',
                        'text': 'Test outpu...'
                }
        """
        self.setup_session(session)
        url = f'{self.api_server}/{self.endpoint_name}'
        params['client_session_auth_key'] = self.client_session_auth_key
        params['wait_for_result'] = not progress_callback
        result = await self.__fetch_async(url, params, request_error_callback)
        if progress_callback:
            if result.get('success'):
                job_id = result['job_id']
                init_progress_info = {
                    'job_id': job_id, 
                    'progress': 0, 
                    'queue_position': -1, 
                    'estimate': -1
                }
                await if_async_else_run(progress_callback, init_progress_info, None)

                if progress_stream:
                    raise NotImplementedError()
                    await self.__do_progress_stream(result_callback, progress_callback)
                else:
                    return await self.__finish_api_request_while_receiving_progress_async(
                        job_id,
                        result_callback,
                        progress_callback,
                        progress_error_callback,
                        progress_interval
                    )
            else:
                await if_async_else_run(result_callback, result)
        else:
            result = self.__convert_result_params(result)
            await if_async_else_run(result_callback, result)

        return result


    def do_api_request(
        self,
        params,
        progress_callback=None,
        progress_error_callback=None,
        progress_interval=DEFAULT_PROGRESS_INTERVAL,
        progress_stream=None
        ):
        """
        Do an synchronous API request with optional progress data via callbacks. 

        Args:
            params (dict): Dictionary with parameters for the the API request like 'prompt' or 'image'
            progress_callback (callable, optional): Callback function or coroutine with arguments  progress_info (dict) and 
                progress_data (dict) for receiving progress data. Defaults to None.
            progress_error_callback (callable, optional): Callback function or coroutine with argument error_description (str) for catching 
                progress errors with successful initial request. Prevents ConnectionErrors during Transmitting. Defaults to None.
            progress_interval (int, optional): Interval in seconds at which progress is checked. Default is 300.
            progress_stream (int, optional): Not implemented yet

        Raises:
            ConnectionError: Raised if client couldn't connect with API server. Also raised if client lost connection during transmitting
                and no progress_error_callback given.
            PermissionError: Raised if client is not logged in the API server
            NotImplementedError: If progress_stream != None

        Returns:
            dict: Dictionary with request result parameters.

        Examples:

            Example job result dict:

            .. highlight:: python
            .. code-block:: python

                result = {
                    'auth': '<name_of_worker>',
                    'compute_duration': 2.4,
                    'images': 'data:image/PNG;base64,...'
                    'job_id': 'JID3',
                    'seed': 1234413214,
                    'success': True,
                    'text': 'Test output...',
                    'total_duration': 2
                }

                
            Example progress_callback arguments progress_info and progress_data dictionaries:          

            .. highlight:: python
            .. code-block:: python

                progress_info = {
                    'job_id': 'JID3', 
                    'progress': 50, 
                    'queue_position': 0, 
                    'estimate': -1
                }
                progress_data = {
                        'info': '<infos from worker about progress', 
                        'images': 'data:image/PNG;base64,...',
                        'text': 'Test outpu...'
                }

        """
        url = f'{self.api_server}/{self.endpoint_name}'
        params['client_session_auth_key'] = self.client_session_auth_key
        params['wait_for_result'] = not progress_callback
        result = self.__fetch_sync(url, params)
        if progress_callback:
            if result.get('success'):
                job_id = result['job_id']
                init_progress_info = {
                    'job_id': job_id,
                    'progress': 0,
                    'queue_position': -1,
                    'estimate': -1
                }
                progress_callback(init_progress_info, None)

                if progress_stream:
                    raise NotImplementedError()
                    self.__do_progress_stream(result_callback, progress_callback)
                else:
                    result = self.__finish_api_request_while_receiving_progress_sync(
                        job_id,
                        progress_callback,
                        progress_error_callback,
                        progress_interval
                    )
        else:
            result = self.__convert_result_params(result)

        return result


    def setup_session(self, session):
        """Open a new session if session is

        Args:
            session (aiohttp.ClientSession): Give existing session to ModelApi API to make upcoming requests in given session.
        """        
        if session:
            self.session = session
        elif not self.session or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def close_session(self):
        """
        Close the aiohttp client session saved in ModelApi().session.
        """
        if self.session:
            await self.session.close()


    def __get_data_format_from_byte_string(self, byte_string_data):
        """
        Check the image format of byte string data and return the format.

        Args:
            byte_string_data (bytes): byte string image data.

        Returns:
            str: The detected image format (e.g., "PNG" or "JPG").
        """
        if byte_string_data.lower().startswith(b'\x89'):
            return 'PNG'
        if byte_string_data.lower().startswith(b'\xff\xd8'):
            return 'JPEG'
        else:
            print("Couldn't read image format from_byte_string data. Image format is set to default value 'PNG'.")
            return 'PNG'


    @staticmethod
    def check_if_valid_base64_string(test_string):
        """
        Check if given string is a valid base64-encoded string.

        Args:
            test_string (str): The string to test.

        Returns:
            bool: True if the string is a valid base64-encoded string, False otherwise.
        """
        try:
            body = test_string.split(',')[1] if ',' in test_string else None
            return base64.b64encode(base64.b64decode(body.encode('utf-8'))).decode('utf-8') == body if body else False
        except TypeError:
            return False


    @staticmethod
    def get_version():
        """Parses name and version of API Python Client Interface with pkg_resources

        Returns:
            str: Name and version of API Python Client Interface
        """        
        try:
            version = str(pkg_resources.get_distribution("python_api_client_interface"))
        except pkg_resources.DistributionNotFound:
            import re
            from pathlib import Path
            setup_py = Path(__file__).resolve().parent.parent / 'setup.py'
            with open(setup_py, 'r') as file:                
                version_no = re.search(r"version\s*=\s*'(.*)'\s*,\s*\n", file.read()).group(1)
            version = f'api_client_interface {version_no}'
        return version


    async def __finish_api_request_while_receiving_progress_async(
        self,
        job_id,
        result_callback,
        progress_callback,
        progress_error_callback,
        progress_interval
        ):
        """
        Finish the asynchronous API request while receiving progress data every progress_interval=300 seconds.

        Args:
            job_id (str): ID of related job.
            result_callback (callable or coroutine, optional): Callback function or coroutine with argument result (dict) 
                to handle the API request result. Accepts synchronous functions and asynchronous couroutines. Defaults to None
            progress_callback (callable or coroutine, optional): Callback function or coroutine with arguments progress_info (dict) and 
                progress_data (dict) for tracking progress. Accepts synchronous functions and asynchronous couroutines. Default is None.
            progress_error_callback (callable or coroutine, optional): Callback function or coroutine with arguments error_description (str) for catching 
                progress errors with successful initial request. Accepts synchronous functions and asynchronous couroutines. Defaults to None.
            progress_interval (int): Interval in seconds at which progress is checked.
        """ 
        job_done = False
        while not job_done:         
            progress_result = await self.__fetch_progress_async(job_id, progress_error_callback)
            job_done = progress_result.get('job_state') == 'done'
            progress_info, progress_data = self.__process_progress_result(progress_result)

            if progress_result.get('job_state') != 'canceled':
                await if_async_else_run(progress_callback, progress_info, progress_data)
                await asyncio.sleep(progress_interval)

        await if_async_else_run(result_callback, progress_data)
        return progress_data


    def __finish_api_request_while_receiving_progress_sync(
        self,
        job_id,
        progress_callback,
        progress_error_callback,
        progress_interval
        ):
        """
        Finish the API request while receiving progress data every progress_interval=300 seconds.

        Args:
            job_id (str): ID of related job.
            progress_callback (callback): Callback function with arguments progress_info (dict) and progress_data (dict) 
                for tracking progress.
            progress_error_callback (callable): Callback function with arguments error_description (str) for catching 
                progress errors with successful initial request.
            progress_interval (int): Interval in seconds at which progress is checked.

        Returns:
            dict: Dictionary with job results

        """ 
        job_done = False
        while not job_done:         
            progress_result = self.__fetch_progress_sync(job_id, progress_error_callback)
            job_done = progress_result.get('job_state') == 'done'
            progress_info, progress_data = self.__process_progress_result(progress_result)
            if progress_result.get('job_state') != 'canceled':
                progress_callback(progress_info, progress_data)
                time.sleep(progress_interval)
        return progress_data
        

    def __process_progress_result(self, progress_result):
        """Format received progress results depending on job state and self.output_format.

        Args:
            progress_result (dict): Progress result dictionary received from API server

        Returns:
            dict, dict: Dictionaries progress_info and progress_result {}
        """
        progress_info = {'job_id': progress_result.get('job_id')}
        if progress_result.get('success'):

            progress = progress_result['progress']
            progress_info['progress'] = progress['progress']
            progress_info['queue_position'] = progress['queue_position']
            progress_info['estimate'] = progress.get('estimate')
            progress_data = progress.get('progress_data')

            if progress_result['job_state'] == 'done':
                job_result = progress_result['job_result']
                job_result['job_id'] = progress_info['job_id']
                job_result['success'] = progress_result['success']
                progress_data = job_result

            elif progress_result['job_state'] == 'canceled':
                return progress_info, progress_data

            return progress_info, self.__convert_result_params(progress_data)

        else:
            return progress_info, progress_result


    async def __fetch_async(
        self,
        url,
        params,
        error_callback=None,
        do_post=True
        ):
        """
        Perform an asynchronous HTTP request to the API server. Python objects and byte string params will be converted automatically to base64 string.
        Base64 strings containing in the result will be converted to self.output_format='base64'.

        Args:
            url (str): The URL for the HTTP request.
            params (dict): Parameters for the HTTP request.
            error_callback (callable or coroutine, optional): Callback function or coroutine with argument error_description (str) for catching 
                errors. Accepts synchronous functions and asynchronous couroutines. Defaults to None.
            do_post (bool, optional): Whether to use a POST request. Defaults to True.

        Returns:
            dict: The result from the API worker via API server.

        Examples:

            Example result:
        
            .. highlight:: python
            .. code-block:: python
                
                result = {
                    'success': True,
                    'job_id': 'JID21',
                    'images': ['data:image/JPEG;base64,/9j/4AA...'],
                    'text': 'Test output...',
                    'seed': 26262303,
                    'prompt': 'cat',
                    'compute_duration': 8.2,
                    'total_duration': 47.8,
                    'auth': 'neo07_NVIDIA A100-SXM4-40GB_0',
                    'worker_interface_version': 'API-Worker-Interface 0.3.5'
                }
        """
        params = self.__convert_object_or_byte_string_params_to_base64(params)
        try:       
            method = self.session.post if do_post else self.session.get
            request_params = {'json': params} if do_post else {'params': params}

            async with method(url, **request_params) as response:
                response_json = await response.json()
                if response.status == 200:
                    return response_json
                else:
                    return await self.__error_handler_async(response, 'API request', error_callback)
                        
        except aiohttp.client_exceptions.ClientConnectorError as error:
            return await self.__error_handler_async(error, 'API request', error_callback)


    def __fetch_sync(
        self,
        url,
        params,
        do_post=True
        ):
        """
        Perform a synchronous HTTP request to the API server. Python objects and byte string params will be converted automatically to base64 string.
        Base64 strings containing in the result will be converted back to python objects or to byte strings.

        Args:
            url (str): The URL for the HTTP request.
            params (dict): Parameters for the HTTP request.
            do_post (bool, optional): Whether to use a POST request. Defaults to True.

        Returns:
            dict: The result from the API worker via API server.

        Examples:

            Example result:
        
            .. highlight:: python
            .. code-block:: python
                
                result = {
                    'success': True,
                    'job_id': 'JID21',
                    'images': ['data:image/JPEG;base64,/9j/4AA...'],
                    'text': 'Test output...',
                    'seed': 26262303,
                    'prompt': 'cat',
                    'compute_duration': 8.2,
                    'total_duration': 47.8,
                    'auth': 'neo07_NVIDIA A100-SXM4-40GB_0',
                    'worker_interface_version': 'API-Worker-Interface 0.3.5'
                }
        """
        params = self.__convert_object_or_byte_string_params_to_base64(params)
        try:
            method = requests.post if do_post else requests.get
            request_params = {'json': params} if do_post else {'params': params}

            response = method(url, **request_params)
            

            if response.status_code == 200:
                return response.json()
            else:
                return self.__error_handler_sync(response, 'API request')
      
        except requests.exceptions.ConnectionError as error:
            return self.__error_handler_sync(error, 'API request')


    async def __fetch_auth_key_async(self, error_callback):
        """
        Asynchronous retrieve of client session authentication key via route /get_client_session_auth_key.

        Args:
            error_callback (callable or coroutine, optional): Callback function or coroutine with argument error_description (str) for catching 
                errors. Accepts synchronous functions and asynchrouns couroutines. Defaults to None.

        Returns:
            str: The client session authentication key.
            
        """
        url = f'{self.api_server}/get_client_session_auth_key'
        params = {
            'endpoint_name': self.endpoint_name, 
            'version': ModelAPI.get_version()
        }
        try:
            async with self.session.get(url=url, params=params) as response:
                response_json = await response.json()
                if response.status == 200:
                    return response_json.get('client_session_auth_key')
                else:
                    return await self.__error_handler_async(response, 'login', error_callback)

        except aiohttp.client_exceptions.ClientConnectorError as error:
            return await self.__error_handler_async(error, 'login', error_callback)


    def __fetch_auth_key(self):
        """
        Synchronous retrieve of client session authentication key via route /get_client_session_auth_key.

        Returns:
            str: The client session authentication key.
        """
        url = f'{self.api_server}/get_client_session_auth_key'
        params = {
            'endpoint_name': self.endpoint_name,
            'version': ModelAPI.get_version()
        }
        try:
            response = requests.get(url=url, params=params)

            if response.status_code == 200:
                return response.json().get('client_session_auth_key')
            else:
                return self.__error_handler_sync(response, 'login', None)

        except requests.exceptions.ConnectionError as error:
            return self.__error_handler_sync(error, 'login', None)


    async def __fetch_progress_async(self, job_id, progress_error_callback):
        """
        Fetch progress data asynchronously from API server for running job with given job id.

        Args:
            job_id (str): Job id of running job
            progress_error_callback (callable or coroutine): Callback function with arguments error_description (str) for catching 
                progress errors with successful initial request. Accepts synchronous functions and asynchrouns couroutines.

        Returns:
            dict: Dictionary with progress result of the job.

        Examples:

            Example progress result dictionary at start:

            .. highlight:: python
            .. code-block:: python

                progress_result = {
                    'job_id': 'JID6',
                    'job_state': 'processing',
                    'progress': {
                        'progress': 0, 
                        'queue_position': 0
                    },
                    'success': True
                }

            Example progress result dictionary while processing:

            .. highlight:: python
            .. code-block:: python

                progress_result = {
                    'job_id': 'JID6',
                    'job_state': 'processing',
                    'progress': {
                        'job_id': 'JID6', 
                        'progress': 50,
                        'progress_data': {
                            'images': ['base64-string', 'base64-string', ...]
                            'text': 'Test output'
                        },
                        'queue_position': 0
                    },
                    'success': True
                }

            Example progress_result dictionaries when finished:

            .. highlight:: python
            .. code-block:: python

                progress_result = {
                    'job_id': 'JID6',
                    'job_result': {
                        'auth': 'worker_name',
                        'compute_duration': 2.4,
                        'images': ['base64-string', 'base64-string', ...]
                        'text': 'Test outpu...',
                        'total_duration': 2.5,
                        'worker_interface_version': 'API-Worker-Interface 0.3.5'
                    },
                    'job_state': 'done',
                    'progress': {
                        'job_id': 'JID6',
                        'progress': 100,
                        'progress_data': {
                            'images': ['base64-string', 'base64-string', ...]
                            'text': 'Test outpu...'
                        },
                        'queue_position': 0
                    },
                    'success': True
                }
        """
        try:
            url = f'{self.api_server}/{self.endpoint_name}/progress'
            params = {
                'client_session_auth_key': self.client_session_auth_key, 
                'job_id': job_id
            }
            async with self.session.get(url, params=params) as response:
                response_json = await response.json()
                if response.status == 200:
                    return response_json
                else:
                    return await self.__error_handler_async(response, 'progress', progress_error_callback)

        except aiohttp.client_exceptions.ClientConnectorError as error:
            return await self.__error_handler_async(response, 'progress', progress_error_callback)


    def __fetch_progress_sync(self, job_id, progress_error_callback):
        """
        Fetch progress data from API server for running job with given job id.

        Args:
            job_id (str): Job id of running job.
            progress_error_callback (callable): Callback function with arguments error_description (str) for catching 
                errors during transmission with successful initial request.

        Returns:
            dict: Progress information for the job.

        Examples:

            Example progress_result dictionaries at start:

            .. highlight:: python
            .. code-block:: python

                progress_result = {
                    'job_id': 'JID6',
                    'job_state': 'processing',
                    'progress': {
                        'progress': 0, 
                        'queue_position': 0
                    },
                    'success': True
                }

            Example progress_result dictionaries while processing:

            .. highlight:: python
            .. code-block:: python

                progress_result = {
                    'job_id': 'JID6',
                    'job_state': 'processing',
                    'progress': {
                        'job_id': 'JID6', 
                        'progress': 50,
                        'progress_data': {
                            'info': 'infos from worker about progress',
                            'images': 'base64-string',
                            'text': 'Test output'
                        },
                        'queue_position': 0
                    },
                    'success': True
                }

            Example progress_result dictionaries when finished:

            .. highlight:: python
            .. code-block:: python

                progress_result = {
                    'job_id': 'JID6',
                    'job_result': {
                        'auth': 'neo07_GPU0',
                        'compute_duration': 2.4,
                        'images': 'data:image...',
                        'text': 'Test outpu...',
                        'total_duration': 2.5,
                        'worker_interface_version': 'API-Worker-Interface 0.3.5'
                    },
                    'job_state': 'done',
                    'progress': {
                        'job_id': 'JID6',
                        'progress': 100,
                        'progress_data': {
                            'info': 'infos from worker about progress',
                            'images': 'data:image...',
                            'text': 'Test outpu...'
                            },
                        'queue_position': 0
                    },
                    'success': True
                }
        """
        try:
            url = f'{self.api_server}/{self.endpoint_name}/progress'
            params = {
                'client_session_auth_key': self.client_session_auth_key, 
                'job_id':job_id
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                return self.__error_handler_sync(response, 'progress', progress_error_callback)

        except requests.exceptions.ConnectionError as error:
            return self.__error_handler_sync(error, 'progress', progress_error_callback)


    async def __error_handler_async(
        self,
        response,
        request_type,
        error_callback
        ):
        """Asynchronous error handler. Calls error_callback if given with error_description as argument, else raises ConnectionError.

        Args:
            response (requests.models.Response): Response of http request
            request_type (str): Type of request (login, API request, progress)
            error_callback (callable or coroutine): Callback function with argument error_description (str) for catching errors.
                Accepts synchronous functions and asynchrouns couroutines.

        Raises:
            ConnectionError: ConnectionError with error description.
            PermissionError: Raised if client is not logged in to the API server.

        Returns:
            str: Error description
        """
        if self.session and not self.session.close:
            await self.session.close()
        status_code = response.status if hasattr(response, 'status') else None
        response_json = await response.json() if status_code else None
            
        error_description = self.__make_error_description(status_code, response_json, request_type)
        
        if error_callback:
            await if_async_else_run(error_callback, error_description)
            return error_description
        elif response_json == {'success': False, 'errors': ['Client session authentication key not registered in API Server']}:
            raise ConnectionRefusedError('Login failed! You first need to run do_login() to login to the API server!\n'+error_description)
        elif request_type == 'progress':
            raise BrokenPipeError('Lost connection while receiving progress. To catch this error, use progress_error_callback')
        else:
            raise ConnectionError(error_description)


    def __error_handler_sync(
        self,
        response,
        request_type,
        error_callback=None
        ):
        """Error handler. Calls error_callback if given with error_description as argument, else raises ConnectionError.

        Args:
            response (requests.models.Response): Response of http request
            request_type (str): Type of request (login, API request, progress)
            error_callback (callable, optional): Callback function with argument error_description (str) for catching errors.
                Defaults to None

        Raises:
            ConnectionError: ConnectionError with error description
            PermissionError: Raised if client is not logged in to the API server.

        Returns:
            str: Error description
        """        
        status_code = response.status_code if hasattr(response, 'status_code') else None
        response_json = response.json() if status_code is not None else None
        error_description = self.__make_error_description(status_code, response_json, request_type)
        
        if error_callback:
            error_callback(error_description)
            return error_description
        elif response_json == {'success': False, 'errors': ['Client session authentication key not registered in API Server']}:
            raise ConnectionRefusedError('Login failed! You first need to run do_login() to login to the API server!\n'+error_description)
        elif request_type == 'progress':
            raise BrokenPipeError('Lost connection while receiving progress. To catch this error, use progress_error_callback')
        else:
            raise ConnectionError(error_description)

        

    def __make_error_description(
        self, 
        status_code, 
        response_json, 
        request_type
        ):
        """Helper method to create error report string

        Args:
            status_code (int): Status code of request response.
            response_json (dict): Dictionary with request response.
            request_type (str): Type of request (login, API request, progress)

        Returns:
            str: Error description
        """        
        status_code_str = f'Status code: {status_code}\nResponse: {response_json}' if status_code else f'Connection to {self.api_server} offline'
        return f'{request_type.capitalize()} at {self.api_server} failed!\n{status_code_str}'


    def __convert_result_params(self, params):
        """
        Converts base64-encoded parameters to byte string data in given dictionary.

        Args:
            params (dict): Dictionary of parameters.

        Returns:
            dict: Dictionary with base64-encoded parameters converted to output format defined in self.output_format.
        """
        params_converted = dict()
        if params:
            for key, value in params.items():
                if isinstance(value, list):
                    value = [self.__convert_base64_to_desired_format(base64_string) for base64_string in value]
                elif isinstance(value, str):
                    value = self.__convert_base64_to_desired_format(value)
                    
                params_converted[key] = value

        return params_converted


    def __convert_base64_to_desired_format(self, base64_string):
        """Convert given base64 string to python object unpacked with pickle.loads().

        Args:
            base64_string (str): Base64 string to be converted

        Raises:
            TypeError: When data couldn't be unpacked.

        Returns:
            str, bytes or object: Base64 string bytes string or python object, depending on self.output_format.
        """
        if ModelAPI.check_if_valid_base64_string(base64_string) and (self.output_format == 'byte_string' or self.output_format == 'object'):
            output = base64.b64decode(base64_string.split(',')[1].encode('utf-8'))
            if self.output_format == 'object':
                try:
                    output = pickle.loads(output)
                except pickle.UnpicklingError:
                    raise TypeError('Error unpacking received data. Change output_format in ModelAPI()')
        else:
            output = base64_string

        return output


    def __convert_object_or_byte_string_params_to_base64(self, params, data_type='image'):
        """
        Convert byte string data parameters to base64 encoding in a dictionary.

        Args:
            params (dict): Dictionary of parameters.

        Returns:
            dict: Dictionary with byte string data parameters converted to base64 encoding.
        """
        data_format = params.pop('image_format') if 'image_format' in params else 'PNG'

        if params:
            for key, value in params.items():
                if isinstance(value, bytes):
                    if not data_format:
                        data_format = self.__get_data_format_from_byte_string(value)
                    params[key] = f'data:{data_type}/{data_format};base64,' + base64.b64encode(value).decode('utf-8')
                elif not isinstance(value, (str, int, float, list, tuple, dict, set)):
                    params[key] = params[key] = f'data:{data_type}/{data_format};base64,' + base64.b64encode(pickle.dumps(value)).decode('utf-8')

        return params




    def __do_progress_stream(self, result_callback, progress_callback):
        """
        Stream progress updates for running API request and handle progress events.
        (Old untested implementation, functionality not guaranteed)

        Args:
            result_callback (callback): Callback function with job_result (dict) as argument to handle the API request result.
            progress_callback (callback, optional): Callback function with arguments progress_info (dict) and progress_data (dict) for tracking progress. Defaults to None.
        """
        async def on_progress(event):
            print('OnProgress:', event.data)
            data = json.loads(event.data)
            progress_info = {
                'progress': data['progress'],
                'queue_position': data['queue_position'],
                'estimate': -1
                            }
            progress_callback(progress_info, data['progress_data'])
            if data['job_state'] == "done":
                await if_async_else_run(result_callback, data['job_result'])

        event_source = self.session.ws_connect(
            f'/stream_progress?client_session_auth_key={self.client_session_auth_key}'
                                                        )
        event_source.receive_json = on_progress


async def do_api_request_async(
    api_server, 
    endpoint_name, 
    params, 
    result_callback = None, 
    progress_callback = None,
    request_error_callback = None,
    progress_error_callback = None,
    session = None
    ):
    """
    A simplified interface for making a single asynchronous API request with do_api_login included.

    Args:
        api_server (str): The address of the API server.
        endpoint_name (str): The name of the API endpoint.
        params (dict): Parameters for the API request.
        result_callback (callback, optional): Callback function with argument result (dict) to handle the API request result. Defaults to None.
        progress_callback (callback, optional): Callback function with arguments progress_info (dict). Defaults to None.
            and progress_data (dict) for tracking progress. Defaults to None.
        request_error_callback (callback, optional): Callback function with arguments error_description (str) 
            for catching request errors. Defaults to None.
        progress_error_callback (callable or coroutine, optional): Callback function with arguments error_description (str) for catching 
            progress errors with successful initial request. Accepts synchronous functions and asynchrouns couroutines. Defaults to None.
        session (aiohttp.ClientSession): Give existing session to ModelApi API to make login request in given session. Defaults to None.

    Returns:
        dict: Dictionary with request result parameters.

    Raises:
        ConnectionError: Raised if client couldn't connect with API server and no request_error_callback is given. Also raised if client lost connection during transmitting
            and no progress_error_callback is given.

    Examples:

        Example usage with synchronous callbacks:

        .. highlight:: python
        .. code-block:: python

            import asyncio

            def result_callback(result):
                process_result(result)

            def progress_callback(progress_info, progress_data):
                process_progress_info(progress_info)
                process_progress_data(progress_data)


            asyncio.run(do_api_request('https://api.aime.team', 'llama2_chat', {'text': 'Chat question'}), result_callback, progress_callback))

        Example usage with asynchronous callbacks:

        .. highlight:: python
        .. code-block:: python

            import asyncio

            async def result_callback(result):
                await process_result(result)

            async def progress_callback(progress_info, progress_data):
                await process_progress_info(progress_info)
                await process_progress_data(progress_data)


            result = asyncio.run(do_api_request('https://api.aime.team', 'llama2_chat', {'text': 'Chat question'}), result_callback, progress_callback))


        Example progress result dictionary at start:

        .. highlight:: python
        .. code-block:: python

            progress_result = {
                'job_id': 'JID6',
                'job_state': 'processing',
                'progress': {
                    'progress': 0, 
                    'queue_position': 0
                },
                'success': True
            }

        Example progress result dictionary while processing:

        .. highlight:: python
        .. code-block:: python

            progress_result = {
                'job_id': 'JID6',
                'job_state': 'processing',
                'progress': {
                    'job_id': 'JID6', 
                    'progress': 50,
                    'progress_data': {
                        'images': 'base64-string',
                        'text': 'Test output'
                    },
                    'queue_position': 0
                },
                'success': True
            }

        Example progress_result dictionaries when finished:

        .. highlight:: python
        .. code-block:: python

            progress_result = {
                'job_id': 'JID6',
                'job_result': {
                    'auth': 'neo07_GPU0',
                    'compute_duration': 2.4,
                    'images': 'data:image...',
                    'text': 'Test outpu...',
                    'total_duration': 2.5
                },
                'job_state': 'done',
                'progress': {
                    'job_id': 'JID6',
                    'progress': 100,
                    'progress_data': {
                        'images': 'data:image...',
                        'text': 'Test outpu...'
                    },
                    'queue_position': 0
                },
                'success': True
            }
    """

    model_api = ModelAPI(api_server, endpoint_name, session)
    auth_key = await model_api.do_api_login_async()
    result = await model_api.do_api_request_async(
        params,
        result_callback,
        progress_callback,
        request_error_callback,
        progress_error_callback
    )
    await model_api.close_session()
    return result


def do_api_request(
    api_server, 
    endpoint_name, 
    params, 
    progress_callback = None,
    progress_error_callback = None
    ):
    """A simplified interface for making a single synchronous API request with do_api_login included.

    Args:
        api_server (str): Address of API server
        endpoint_name (str): Name of endpoint
        params (dict): Dictionary with api request parameters
        progress_callback (callback, optional): Callback function with arguments progress_info (dict) 
            and progress_data (dict) for tracking progress. Defaults to None.
        progress_error_callback (callback, optional): Callback function with argument error_description (str) 
            called if request was successfull but progress got errors. Defaults to None.

    Returns:
        dict: Dictionary with request result parameters

    Raises:
        ConnectionError: Raised if client couldn't connect with API server. Also raised if client lost connection during transmitting
            and no progress_error_callback is given.


    Examples:

        Example usage with progress and progress_error_callback:

        .. highlight:: python
        .. code-block:: python

            import asyncio

            def progress_callback(progress_info, progress_data):
                process_progress_info(progress_info)
                process_progress_data(progress_data)

            def progress_error_callback(error_description):
                pass

            result = do_api_request('https://api.aime.team', 'llama2_chat', {'text': 'Chat question'}), progress_callback, progress_error_callback)
        
        Example progress result dictionary at start:

        .. highlight:: python
        .. code-block:: python

            progress_result = {
                'job_id': 'JID6',
                'job_state': 'processing',
                'progress': {
                    'progress': 0, 
                    'queue_position': 0
                },
                'success': True
            }

        Example progress result dictionary while processing:

        .. highlight:: python
        .. code-block:: python

            progress_result = {
                'job_id': 'JID6',
                'job_state': 'processing',
                'progress': {
                    'job_id': 'JID6', 
                    'progress': 50,
                    'progress_data': {
                        'images': 'base64-string',
                        'text': 'Test output'
                    },
                    'queue_position': 0
                },
                'success': True
            }

        Example progress_result dictionaries when finished:

        .. highlight:: python
        .. code-block:: python

            progress_result = {
                'job_id': 'JID6',
                'job_result': {
                    'auth': 'neo07_GPU0',
                    'compute_duration': 2.4,
                    'images': 'data:image...',
                    'text': 'Test outpu...',
                    'total_duration': 2.5
                },
                'job_state': 'done',
                'progress': {
                    'job_id': 'JID6',
                    'progress': 100,
                    'progress_data': {
                        'images': 'data:image...',
                        'text': 'Test outpu...'
                    },
                    'queue_position': 0
                },
                'success': True
            }
    """ 
    model_api = ModelAPI(api_server, endpoint_name)
    client_session_auth_key = model_api.do_api_login()
    return model_api.do_api_request(params, progress_callback, progress_error_callback)
    
    
async def if_async_else_run(callback, *args):    
    """Helper method to either await asynchronous coroutine or call synchronous functions.

    Args:
        callback (function or coroutine): Await asynchronous coroutine, call synchronous functions.

    Returns:
        callback(\*args): Result of callback
    """    
    if asyncio.iscoroutinefunction(callback):
        return await callback(*args)
    elif callable(callback):
        return callback(*args)

