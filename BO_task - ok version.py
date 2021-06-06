from clearml import Task
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
import pandas as pd
from clearml.automation import (
    DiscreteParameterRange, HyperParameterOptimizer, UniformParameterRange,
    UniformIntegerParameterRange)

try:
    from clearml.automation.optuna import OptimizerOptuna  # noqa
    optunaStrategy = True
except ImportError as ex:
	print("Unable to import Optuna package!")

try:
    from clearml.automation.hpbandster import OptimizerBOHB  # noqa
    bohbStrategy = True
except ImportError as ex:
	print("Unable to import BOHB package!")


######## REQUIRED PARAMETERS FOR BOtask CLASS !!!! #########
# base_task_project_name => the name of the project which the base task resides
# base_task_name => the name of the base task
# params => the hyperparameters to be tuned in the form of a dictionary. Example:
#
# param = {
# 		'layer_1': ('uniformIntegerParameterRange',[128,512,128]),
# 		'layer_2': ('uniformIntegerParameterRange',[128,512,128]),
# 		'optimizer': ('discreteParameterRange',['RMSprop','SGD'])
# 	}
#
# metric => metric to be used to optimize model, can only be a string of either 'loss' or 'accuracy'
# hyperparameters_section => the section in which the hyperparameters reside in the UI, usually 'General' or if `argparse` is used in base experiment 'Args' is used
# objective_metric_title => this is the title of the scalar objective metric we want to maximize/minimize in the base experiment
# objective_metric_series => this is the series of the scalar objective metric we want to maximize/minimize in the base experiment


class BOtask:
	def __init__(self, base_task_project_name, base_task_name, params, metric, hyperparameters_section, objective_metric_title, objective_metric_series, tuner_project_name='hyperparameter_tuning', tuner_task_name='hyperparameter tuning', optimizer='BayesianOptimization', max_number_of_concurrent_tasks=1,execution_queue='default',time_limit_per_job=10., pool_period_min=0.2, total_max_jobs=10, min_iteration_per_job=3,max_iteration_per_job=6):
		self.task = Task.init(project_name=tuner_project_name, task_name=tuner_task_name, task_type=Task.TaskTypes.optimizer)
		self.logger = self.task.get_logger()
		self.base_task_project = base_task_project_name
		self.base_task_name = base_task_name
		self.template_task = Task.get_task(project_name=self.base_task_project, task_name=self.base_task_name)
		self.params = params
		self.metric = metric
		if self.metric == 'loss':
			self.metric_sign = 'min'
		else:
			self.metric_sign = 'max'
		self.optimizer = optimizer
		self.hyperparameters_section = hyperparameters_section
		self.objective_metric_title = objective_metric_title
		self.objective_metric_series = objective_metric_series
		self.max_number_of_concurrent_tasks = max_number_of_concurrent_tasks
		self.execution_queue = execution_queue
		self.time_limit_per_job = time_limit_per_job
		self.pool_period_min = pool_period_min
		self.total_max_jobs = total_max_jobs
		self.min_iteration_per_job = min_iteration_per_job
		self.max_iteration_per_job = max_iteration_per_job
		self.bo_params = {}
		self.clearml_params = []
		self.iteration = 0
		self.columns = []
		self.df = pd.DataFrame()
		self.df_best = pd.DataFrame()

	def start(self):
		self.handle_params()
		if(self.optimizer=='Optuna' and optunaStrategy):
			self.Optuna_tuner()
		elif(self.optimizer=='BOHB' and bohbStrategy):
			self.BOHB_tuner()
		elif(self.optimizer=='BayesianOptimization'):
			self.BO_tuner()
		else:
			self.BO_tuner()


	def handle_params(self):

		for k in self.params.keys():
			if self.params[k][0] == 'discrete':
				self.bo_params[k] = tuple(self.params[k][1])
			elif self.params[k][0] == 'categorical':
				self.bo_params[k] = (0,len(self.params[k][1])-1)
				self.clearml_params.append(DiscreteParameterRange(self.hyperparameters_section+'/'+k, values=self.params[k][1]))
			elif self.params[k][0] == 'continuous':
				self.bo_params[k] = tuple(self.params[k][1])
			elif self.params[k][0] == 'uniformIntegerParameterRange':
				self.params[k] = ('categorical',self.params[k][1])
				self.clearml_params.append(UniformIntegerParameterRange(self.hyperparameters_section+'/'+k,min_value=self.params[k][1][0], max_value=self.params[k][1][1], step_size=self.params[k][1][2]))
				no_of_steps = int((self.params[k][1][1]-self.params[k][1][0])/self.params[k][1][2])
				values = [self.params[k][1][0]]
				for i in range(1,no_of_steps+1):
					values.append(self.params[k][1][0]+i*self.params[k][1][2])
				self.params[k] = (self.params[k][0], values)
				self.bo_params[k] = (0,len(self.params[k][1])-1)
			elif self.params[k][0] == 'uniformParameterRange':
				self.params[k] = ('categorical',self.params[k][1])
				self.clearml_params.append(UniformParameterRange(self.hyperparameters_section+'/'+k,min_value=self.params[k][1][0], max_value=self.params[k][1][1], step_size=self.params[k][1][2]))
				no_of_steps = int((self.params[k][1][1]-self.params[k][1][0])/self.params[k][1][2])
				values = [self.params[k][1][0]]
				for i in range(1,no_of_steps+1):
					values.append(self.params[k][1][0]+float(i)*self.params[k][1][2])
				self.params[k] = (self.params[k][0], values)
				self.bo_params[k] = (0,len(self.params[k][1])-1)
			elif self.params[k][0] == 'discreteParameterRange':
				self.params[k] = ('categorical',self.params[k][1])
				self.clearml_params.append(DiscreteParameterRange(self.hyperparameters_section+'/'+k, values=self.params[k][1]))
				self.bo_params[k] = (0,len(self.params[k][1])-1)


		print(self.params)
		print(self.bo_params)
		print(self.clearml_params)

		columns = []

		for k in self.params.keys():
			columns.append(k)

		columns.append(self.metric)
		columns.append('epoch')

		self.columns = columns

		self.df = pd.DataFrame(columns=columns)
		self.df_best = pd.DataFrame(columns=columns)

	def job_complete_callback(
		task_object,
		job_id,                 # type: str
		objective_value,        # type: float
		objective_iteration,    # type: int
		job_parameters,         # type: dict
		top_performance_job_id  # type: str
	):
		try:
			print("task_object: " + str(task_object))
			task_object.logger.report_scalar(title='validation_data',series=task_object.objective_metric_series, value=Task.get_task(task_id=job_id).get_last_scalar_metrics()[task_object.objective_metric_title][task_object.objective_metric_series]['last'], iteration=task_object.iteration)
			# task_object.logger.report_scalar(title='validation_data',series='loss', value=Task.get_task(task_id=job_id).get_last_scalar_metrics()[task_object.objective_metric_title]['loss']['last'], iteration=task_object.iteration)
		except:
			print("Error when printing and logging main task_object's objective_metric last scalar metric.")

		try:
			epochs = Task.get_task(task_id=job_id).get_reported_scalars()[task_object.objective_metric_title][task_object.objective_metric_series]['x']
			objectives = Task.get_task(task_id=job_id).get_reported_scalars()[task_object.objective_metric_title][task_object.objective_metric_series]['y']
			objective = objectives[0]
			epoch = 0
			for i in range(1, len(objectives)):
				if(task_object.metric == 'loss'):
					if(objectives[i]<objective):
						objective = objectives[i]
						epoch = i
				elif(task_object.metric == 'accuracy'):
					if(objectives[i]>objective):
						objective = objectives[i]
						epoch = i
			print("Best " + task_object.metric + ": " + str(objective))
			print("Obtained at " + str(epoch) + " epoch")

			row = []

			for key, value in job_parameters.items():
				row.append(value)

			row = row[:-1]
			row.append(objective)
			row.append(epoch)

			task_object.df_best.loc[task_object.iteration] = row

			df = pd.DataFrame(columns=task_object.columns)
			df.loc[0] = row

			task_object.logger.report_table(title=job_id,series="Best " + task_object.objective_metric_series + ' DataFrame',iteration=task_object.iteration,table_plot=df)

		except:
			print("Unable to record best objective metric at a particular epoch")

		try:
			epoch = objective_iteration
			objective = Task.get_task(task_id=job_id).get_last_scalar_metrics()[task_object.objective_metric_title][task_object.objective_metric_series]['last']

			row = []

			for key, value in job_parameters.items():
				row.append(value)

			row = row[:-1]
			row.append(objective)
			row.append(epoch)

			task_object.df.loc[task_object.iteration] = row
			
		except:
			print("Unable to log any metric.")

		task_object.iteration += 1

		print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
		if job_id == top_performance_job_id:
		    print('WOOT WOOT we broke the record! Objective reached {}'.format(objective_value))

	def Optuna_tuner(self):

		an_optimizer = HyperParameterOptimizer(
		# This is the experiment we want to optimize
		base_task_id=self.template_task.id,
		# here we define the hyper-parameters to optimize
		# Notice: The parameter name should exactly match what you see in the UI: <section_name>/<parameter>
		# For Example, here we see in the base experiment a section Named: "General"
		# under it a parameter named "batch_size", this becomes "General/batch_size"
		# If you have `argparse` for example, then arguments will appear under the "Args" section,
		# and you should instead pass "Args/batch_size"
		hyper_parameters=self.clearml_params,
		# this is the objective metric we want to maximize/minimize
		objective_metric_title=self.objective_metric_title,
		objective_metric_series=self.objective_metric_series,
		# now we decide if we want to maximize it or minimize it (accuracy we maximize)
		objective_metric_sign=self.metric_sign,
		# let us limit the number of concurrent experiments,
		# this in turn will make sure we do dont bombard the scheduler with experiments.
		# if we have an auto-scaler connected, this, by proxy, will limit the number of machine
		max_number_of_concurrent_tasks=self.max_number_of_concurrent_tasks,
		# this is the optimizer class (actually doing the optimization)
		# Currently, we can choose from GridSearch, RandomSearch or OptimizerBOHB (Bayesian optimization Hyper-Band)
		# more are coming soon...
		optimizer_class=OptimizerOptuna,
		# Select an execution queue to schedule the experiments for execution
		execution_queue=self.execution_queue,
		# If specified all Tasks created by the HPO process will be created under the `spawned_project` project
		spawn_project=None,  # 'HPO spawn project',
		# If specified only the top K performing Tasks will be kept, the others will be automatically archived
		save_top_k_tasks_only=None,  # 5,
		# Optional: Limit the execution time of a single experiment, in minutes.
		# (this is optional, and if using  OptimizerBOHB, it is ignored)
		time_limit_per_job=self.time_limit_per_job,
		# Check the experiments every 12 seconds is way too often, we should probably set it to 5 min,
		# assuming a single experiment is usually hours...
		pool_period_min=self.pool_period_min,
		# set the maximum number of jobs to launch for the optimization, default (None) unlimited
		# If OptimizerBOHB is used, it defined the maximum budget in terms of full jobs
		# basically the cumulative number of iterations will not exceed total_max_jobs * max_iteration_per_job
		total_max_jobs=self.total_max_jobs,
		# set the minimum number of iterations for an experiment, before early stopping.
		# Does not apply for simple strategies such as RandomSearch or GridSearch
		min_iteration_per_job=self.min_iteration_per_job,
		# Set the maximum number of iterations for an experiment to execute
		# (This is optional, unless using OptimizerBOHB where this is a must)
		max_iteration_per_job=self.max_iteration_per_job,
		)
		# report every 12 seconds, this is way too often, but we are testing here J
		an_optimizer.set_report_period(2.2)
		# start the optimization process, callback function to be called every time an experiment is completed
		# this function returns immediately
		an_optimizer.start(job_complete_callback=self.job_complete_callback)
		# set the time limit for the optimization process (2 hours)
		an_optimizer.set_time_limit(in_minutes=120.0)
		# wait until process is done (notice we are controlling the optimization process in the background)
		an_optimizer.wait()
		# optimization is completed, print the top performing experiments id
		top_exp = an_optimizer.get_top_experiments(top_k=3)
		print([t.id for t in top_exp])
		# make sure background optimization stopped
		an_optimizer.stop()

		self.logger.report_table(title='tuning experiments',series='tuning DataFrame',iteration=0,table_plot=self.df)
		self.logger.report_table(title='tuning experiments best',series='best metrics at particular epoch DataFrame',iteration=0,table_plot=self.df_best)

		print('We are done, good bye')

	def BOHB_tuner(self):

		an_optimizer = HyperParameterOptimizer(
		# This is the experiment we want to optimize
		base_task_id=self.template_task.id,
		# here we define the hyper-parameters to optimize
		# Notice: The parameter name should exactly match what you see in the UI: <section_name>/<parameter>
		# For Example, here we see in the base experiment a section Named: "General"
		# under it a parameter named "batch_size", this becomes "General/batch_size"
		# If you have `argparse` for example, then arguments will appear under the "Args" section,
		# and you should instead pass "Args/batch_size"
		hyper_parameters=self.clearml_params,
		# this is the objective metric we want to maximize/minimize
		objective_metric_title=self.objective_metric_title,
		objective_metric_series=self.objective_metric_series,
		# now we decide if we want to maximize it or minimize it (accuracy we maximize)
		objective_metric_sign=self.metric_sign,
		# let us limit the number of concurrent experiments,
		# this in turn will make sure we do dont bombard the scheduler with experiments.
		# if we have an auto-scaler connected, this, by proxy, will limit the number of machine
		max_number_of_concurrent_tasks=self.max_number_of_concurrent_tasks,
		# this is the optimizer class (actually doing the optimization)
		# Currently, we can choose from GridSearch, RandomSearch or OptimizerBOHB (Bayesian optimization Hyper-Band)
		# more are coming soon...
		optimizer_class=OptimizerBOHB,
		# Select an execution queue to schedule the experiments for execution
		execution_queue=self.execution_queue,
		# If specified all Tasks created by the HPO process will be created under the `spawned_project` project
		spawn_project=None,  # 'HPO spawn project',
		# If specified only the top K performing Tasks will be kept, the others will be automatically archived
		save_top_k_tasks_only=None,  # 5,
		# Optional: Limit the execution time of a single experiment, in minutes.
		# (this is optional, and if using  OptimizerBOHB, it is ignored)
		time_limit_per_job=self.time_limit_per_job,
		# Check the experiments every 12 seconds is way too often, we should probably set it to 5 min,
		# assuming a single experiment is usually hours...
		pool_period_min=self.pool_period_min,
		# set the maximum number of jobs to launch for the optimization, default (None) unlimited
		# If OptimizerBOHB is used, it defined the maximum budget in terms of full jobs
		# basically the cumulative number of iterations will not exceed total_max_jobs * max_iteration_per_job
		total_max_jobs=self.total_max_jobs,
		# set the minimum number of iterations for an experiment, before early stopping.
		# Does not apply for simple strategies such as RandomSearch or GridSearch
		min_iteration_per_job=self.min_iteration_per_job,
		# Set the maximum number of iterations for an experiment to execute
		# (This is optional, unless using OptimizerBOHB where this is a must)
		max_iteration_per_job=self.max_iteration_per_job,
		)
		# report every 12 seconds, this is way too often, but we are testing here J
		an_optimizer.set_report_period(2.2)
		# start the optimization process, callback function to be called every time an experiment is completed
		# this function returns immediately
		an_optimizer.start(job_complete_callback=self.job_complete_callback)
		# set the time limit for the optimization process (2 hours)
		an_optimizer.set_time_limit(in_minutes=120.0)
		# wait until process is done (notice we are controlling the optimization process in the background)
		an_optimizer.wait()
		# optimization is completed, print the top performing experiments id
		top_exp = an_optimizer.get_top_experiments(top_k=3)
		print([t.id for t in top_exp])
		# make sure background optimization stopped
		an_optimizer.stop()

		self.logger.report_table(title='tuning experiments',series='tuning DataFrame',iteration=0,table_plot=self.df)
		self.logger.report_table(title='tuning experiments best',series='best metrics at particular epoch DataFrame',iteration=0,table_plot=self.df_best)

		print('We are done, good bye')

	def BO_tuner(self):

		if self.metric == 'accuracy':
			print("Optimizing model accuracy")
			objective = self.acc_objective
		elif self.metric == 'loss':
			print("Optimizing model loss")
			objective = self.loss_objective

		bayes_op = BayesianOptimization(objective, self.bo_params)

		bayes_op.maximize(n_iter=10, init_points=2)

		dict = bayes_op.max

		for k in dict['params'].keys():
			if self.params[k][0] == 'discrete':
				dict['params'][k] = int(dict['params'][k])
			elif self.params[k][0] == 'categorical':
				dict['params'][k] = self.params[k][1][int(round(dict['params'][k]))]
			elif self.params[k][0] == 'continuous':
				dict['params'][k] = dict['params'][k]

		if self.metric == 'loss':
			dict["target"] = 1.0/dict["target"]

		print(dict)

		self.logger.report_table(title='tuning experiments',series='pandas DataFrame',iteration=0,table_plot=self.df)

		try:
			self.task.upload_artifact('best_params', dict)
		except:
			print("Fail to upload best params dict")

	def acc_objective(self, **data):

		for k in data.keys():
			if self.params[k][0] == 'discrete':
				data[k] = int(data[k])
			elif self.params[k][0] == 'categorical':
				data[k] = self.params[k][1][int(round(data[k]))]
			elif self.params[k][0] == 'continuous':
				data[k] = data[k]

		row = []

		for key, value in data.items():
			print("{} is {}".format(key,value))
			row.append(value)


		cloned_task = Task.clone(source_task=self.template_task,
	                                 name=self.template_task.name+' {}'.format(self.iteration), parent=self.template_task.id)

		# get the original template parameters
		cloned_task_parameters = cloned_task.get_parameters()

		for k in data.keys():
			cloned_task_parameters[k] = data[k]

		cloned_task.set_parameters(cloned_task_parameters)

		# enqueue the task for execution
		Task.enqueue(cloned_task.id, queue_name='default')
		print('Experiment id={} enqueue for execution'.format(cloned_task.id))

		cloned_task.wait_for_status()
		if(cloned_task.get_last_scalar_metrics()):
			print(cloned_task.get_last_scalar_metrics())
			print(cloned_task.get_parameters())
			# self.logger.report_scalar(title='val_loss',series='loss', value=cloned_task.get_last_scalar_metrics()['evaluate']['loss']['last'], iteration=self.iteration)
			self.logger.report_scalar(title='val_acc',series='accuracy', value=cloned_task.get_last_scalar_metrics()[self.objective_metric_title][self.objective_metric_series]['last'], iteration=self.iteration)
			acc = cloned_task.get_last_scalar_metrics()[self.objective_metric_title][self.objective_metric_series]['last']

		row.append(acc)

		self.df.loc[self.iteration] = row

		self.iteration += 1

		return acc

	def loss_objective(self, **data):

		for k in data.keys():
			if self.params[k][0] == 'discrete':
				data[k] = int(data[k])
			elif self.params[k][0] == 'categorical':
				data[k] = self.params[k][1][int(round(data[k]))]
			elif self.params[k][0] == 'continuous':
				data[k] = data[k]

		row = []

		for key, value in data.items():
			print("{} is {}".format(key,value))
			row.append(value)

		cloned_task = Task.clone(source_task=self.template_task,
	                                 name=self.template_task.name+' {}'.format(self.iteration), parent=self.template_task.id)

		# get the original template parameters
		cloned_task_parameters = cloned_task.get_parameters()

		for k in data.keys():
			cloned_task_parameters[k] = data[k]

		cloned_task.set_parameters(cloned_task_parameters)

		# enqueue the task for execution
		Task.enqueue(cloned_task.id, queue_name='default')
		print('Experiment id={} enqueue for execution'.format(cloned_task.id))

		cloned_task.wait_for_status()
		if(cloned_task.get_last_scalar_metrics()):
			print(cloned_task.get_last_scalar_metrics())
			print(cloned_task.get_parameters())
			self.logger.report_scalar(title='val_loss',series='loss', value=cloned_task.get_last_scalar_metrics()[self.objective_metric_title][self.objective_metric_series]['last'], iteration=self.iteration)
			# self.logger.report_scalar(title='val_acc',series='accuracy', value=cloned_task.get_last_scalar_metrics()['evaluate']['accuracy']['last'], iteration=self.iteration)
			loss = cloned_task.get_last_scalar_metrics()[self.objective_metric_title][self.objective_metric_series]['last']

		row.append(loss)

		self.df.loc[self.iteration] = row

		self.iteration += 1

		return 1.0/loss



# param = {
# 		'layer_1': ('categorical',[128,256,512]),
# 		'layer_2': ('categorical',[128,256,512]),
# 		'optimizer': ('categorical',['RMSprop','SGD'])
# 	}

# param = {
# 		'layer_1': ('uniformIntegerParameterRange',[128,512,128]),
# 		'layer_2': ('uniformIntegerParameterRange',[128,512,128]),
# 		'optimizer': ('discreteParameterRange',['RMSprop','SGD'])
# 	}

param = {
		'batch_size': ('uniformIntegerParameterRange',[32,64,16]),
		'lr': ('uniformParameterRange',[0.005,0.015,0.005]),
		'momentum': ('discreteParameterRange',[0.0,0.5])
	}

# bayes = BOtask(base_task_project_name='BO_tuning_3', base_task_name='base task', params=param,metric='loss', hyperparameters_section='General', objective_metric_title='epoch_loss', objective_metric_series='validation: epoch_loss',tuner_project_name='BO_tuning_3', tuner_task_name='hyperparameter tuning', optimizer='BOHB')
bayes = BOtask(base_task_project_name='BO_tuning_3', base_task_name='pytorch base', params=param,metric='loss', hyperparameters_section='Args', objective_metric_title='test', objective_metric_series='loss',tuner_project_name='BO_tuning_3', tuner_task_name='pytorch hyperparameter tuning', optimizer='BOHB')
bayes.start()