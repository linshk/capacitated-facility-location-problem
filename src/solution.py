# _*_ coding: utf-8 _*_
import sys
import numpy
import copy
import random
import math
import matplotlib.pyplot as plt
import pandas as pd
import time
# 存储问题相关的变量
global ficility_count
global customer_count
global capacities
global opening_costs
global demands
global assignment_costs_matrix

# 存储当前的解
global ficilities
global customers
global total_cost

def is_number(str):
	try:
		num = int(str)
		return True
	except:
		return False

class Customer:
	'需求点'
	def __init__(self, cid, demand):
		self.id = cid
		self.demand = demand
		self.fid = -1
		self.assignment_cost = -1

	def __str__(self):
		return 'Customer(id:%d, demand:%d, fid:%d, cost:%d)' % (self.id, self.demand, self.fid, self.assignment_cost)


class Ficility:
	'服务站'
	def __init__(self, fid, capacity, opening_cost, assignment_costs):
		self.id = fid
		self.capacity = capacity
		self.opening_cost = opening_cost
		self.assignment_costs = assignment_costs
		self.assignment = []
		self.cur_capacity = capacity
		self.total_cost = 0

	# 计算将某一需求点分配到该服务站所新增的代价		
	def get_cost(self, customer):
		if self.cur_capacity < customer.demand:
			return float("inf")
		if len(self.assignment) == 0:
			return self.opening_cost + self.assignment_costs[customer.id]
		return self.assignment_costs[customer.id]
	# 将某一需求点分配到该服务站并更新服务站和需求点的状态,返回所需的cost
	def assign(self, customer):
		cost = self.get_cost(customer)
		if cost == float('inf'):
			return 0
		self.total_cost += cost
		self.cur_capacity -= customer.demand
		self.assignment.append(customer)
		customer.fid = self.id
		customer.assignment_cost = self.assignment_costs[customer.id]
		return cost
	# 将某一需求点从该服务站的分配列表中删除并更新服务站和需求点的状态，返回减少的cost
	def deassign(self, customer):
		if customer.fid != self.id:
			return 0
		cost = 0
		for i in range(len(self.assignment)):
			if self.assignment[i].id == customer.id:
				cost += self.assignment[i].assignment_cost
				self.cur_capacity += self.assignment[i].demand
				del self.assignment[i]
				if len(self.assignment) == 0:
					cost += self.opening_cost
				customer.fid = -1
				customer.assignment_cost = -1
				break
		self.total_cost -= cost
		return cost

	def __str__(self):
		tostr = 'Ficility(id-%d, cap-%d, o_cost-%d, c-cap-%d, cost-%d)' % (self.id, self.capacity, self.opening_cost, self.cur_capacity, self.total_cost)
		for customer in self.assignment:
			tostr = tostr + '\n' + str(customer)
		return tostr

def gen_random_index_pair(lowerbound, upperbound):
	x = random.randint(lowerbound, upperbound)
	y = random.randint(lowerbound, upperbound)
	while x == y:
		y = random.randint(lowerbound, upperbound)
	return x, y

# 随机打乱一个列表若干次
def reorder_random(swap_times, li):
	temp = copy.deepcopy(li)
	for i in range(swap_times):
		x, y = gen_random_index_pair(0, len(temp) - 1)
		temp[x], temp[y] = temp[y], temp[x]
	return temp

# 将ficilities或customers列表按id排序
def sort_list(li):
	temp = [None for i in li]
	for i in range(len(li)):
		temp[li[i].id] = li[i]
	return temp

# 随机交换两个需求点，并返回cost的变化(大于0表示cost增加小于0表示减少)，结果保存在传入的参数中
# 扰动程度较小
def get_neighbor_method_1(cur_ficilities, cur_customers, cur_total_cost, max_retries=100):
	global customer_count
	cid1, cid2 = gen_random_index_pair(0, customer_count - 1)
	# 产生新的随机数直至需求点cid1和cid2能够互换位置而不会超出服务站容量
	retries = 0
	while (cur_ficilities[cur_customers[cid1].fid].cur_capacity < cur_customers[cid2].demand - cur_customers[cid1].demand or cur_ficilities[cur_customers[cid2].fid].cur_capacity < cur_customers[cid1].demand - cur_customers[cid2].demand):
		cid1, cid2 = gen_random_index_pair(0, customer_count - 1)
		retries += 1
		# 防止死循环
		if retries > max_retries:
			break
	if retries > max_retries:
		return 0
  
	fid1 = cur_customers[cid1].fid
	fid2 = cur_customers[cid2].fid
	cost1 = cur_ficilities[fid1].deassign(cur_customers[cid1])
	cost2 = cur_ficilities[fid2].deassign(cur_customers[cid2])
	cost3 = cur_ficilities[fid2].assign(cur_customers[cid1])
	cost4 = cur_ficilities[fid1].assign(cur_customers[cid2])
	return (cost3 + cost4) - (cost1 + cost2)

# 随机选取两个需求点，将其中一个需求点分配给另一个需求点所在服务站
# 扰动程度比较小
def get_neighbor_method_2(cur_ficilities, cur_customers, cur_total_cost, max_retries=100):
	global customer_count
	cid1, cid2 = gen_random_index_pair(0, customer_count - 1)
	# 将需求点cid2分配到需求点cid1所在的服务站
	retries = 0
	while cur_ficilities[cur_customers[cid1].fid].cur_capacity < cur_customers[cid2].demand:
		cid1, cid2 = gen_random_index_pair(0, customer_count - 1)
		retries += 1
		# 防止死循环
		if retries > max_retries:
			break
	if retries > max_retries:
		return 0
  
	fid1 = cur_customers[cid1].fid
	fid2 = cur_customers[cid2].fid
	cost1 = cur_ficilities[fid2].deassign(cur_customers[cid2])
	cost2 = cur_ficilities[fid1].assign(cur_customers[cid2])
	return cost2 - cost1


# 将需求点顺序打乱再调用greedy函数通过贪心策略产生另一组解
# 扰动比较大，可帮助模拟退火算法跳出局部最优
def get_neighbor_method_3(cur_ficilities, cur_customers, cur_total_cost):
	global customer_count
	global ficilities
	global customers
	new_customers = reorder_random(customer_count // 2, customers)
	new_ficilities, new_customers, new_total_cost = greedy(ficilities, new_customers)
	# 将结果写入原来的列表
	for i in range(len(new_ficilities)):
		cur_ficilities[i] = new_ficilities[i]
	for i in range(len(new_customers)):
		cur_customers[i] = new_customers[i]
	return new_total_cost - cur_total_cost

# 通过若干种邻域操作获得新解，从中挑选最好的一个并返回
def get_neighbor(cur_ficilities, cur_customers, cur_total_cost):
	new_ficilities_1 = copy.deepcopy(cur_ficilities)
	new_customers_1 = copy.deepcopy(cur_customers)
	cost_diff_1 = get_neighbor_method_1(new_ficilities_1, new_customers_1, cur_total_cost)

	new_ficilities_2 = copy.deepcopy(cur_ficilities)
	new_customers_2 = copy.deepcopy(cur_customers)
	cost_diff_2 = get_neighbor_method_2(new_ficilities_2, new_customers_2, cur_total_cost)

	#new_ficilities_3 = copy.deepcopy(cur_ficilities)
	#new_customers_3 = copy.deepcopy(cur_customers)
	#cost_diff_3 = get_neighbor_method_3(new_ficilities_3, new_customers_3, cur_total_cost)

	min_cost_diff = min(cost_diff_1, cost_diff_2)
	if min_cost_diff == cost_diff_1:
		return new_ficilities_1, new_customers_1, cost_diff_1
	if min_cost_diff == cost_diff_2:
		return new_ficilities_2, new_customers_2, cost_diff_2
	#if min_cost_diff == cost_diff_3:
	#	return new_ficilities_3, new_customers_3, cost_diff_3
	
# 模拟退火算法
def SA(ficilities, customers, init_cost, initial_temp, min_temp, rate, round_times, min_cost):
	cur_ficilities = copy.deepcopy(ficilities)
	cur_customers = copy.deepcopy(customers)
	cur_total_cost = init_cost
	cur_temp = initial_temp
	best_ficilities = None
	best_customers = None
	best_cost = float('inf')
	# 辅助变量，用于跟踪算法的效果以及绘图
	epoch = 0
	# 记录估算精度随温度的变化
	temps = []
	precisions = []
	# 记录整个过程代价的变化
	costs_record = []
	while cur_temp > min_temp:
		# 内循环
		for i in range(round_times):
			new_ficilities, new_customers, cost_diff = get_neighbor(cur_ficilities, cur_customers, cur_total_cost)
			costs_record.append(cur_total_cost + cost_diff)
			# 记录当前找到的最优解
			if cur_total_cost + cost_diff < best_cost:
				best_ficilities = new_ficilities
				best_customers = new_customers
				best_cost = cur_total_cost + cost_diff
			# 若符合条件则接受新解
			if cost_diff < 0 or random.random() < math.exp(-1 * (abs(cost_diff) / cur_temp)):
				cur_ficilities = new_ficilities
				cur_customers = new_customers
				cur_total_cost += cost_diff
		# 降温
		cur_temp = cur_temp * rate
		epoch += 1
		print('当前降温次数，温度，总代价，误差率：NO.%4d %.2f %d %.2f%%' % (epoch, cur_temp, cur_total_cost, 100 * (cur_total_cost - min_cost) / min_cost))
		temps.append(cur_temp)
		precisions.append(100 * (best_cost - min_cost) / min_cost)
	states = [temps, precisions, costs_record]
	return best_ficilities, best_customers, best_cost, states		


# 从文件中加载问题实例
def load_problem(filename):
	global ficility_count
	global customer_count
	global capacities
	global opening_costs
	global demands
	global assignment_costs_matrix
	
	with open(filename, 'r') as f:
		line = f.readline().split()
		index = 0
		ficility_count = int(line[0])
		customer_count = int(line[1])

		capacities = numpy.zeros([ficility_count])
		opening_costs = numpy.zeros([ficility_count])
		demands = numpy.zeros([customer_count])
		assignment_costs_matrix = numpy.zeros([ficility_count, customer_count])

		for i in range(ficility_count):
			line = f.readline().split()
			capacities[i] = int(line[0])
			opening_costs[i] = int(line[1])
			#print(capacities[i], opening_costs[i])

		index = 0
		while index < customer_count:
			line = f.readline().replace('.','').split()
			#print(line)
			for item in line:
				if is_number(item) == False:
					continue
				demands[index] = int(item)
				index += 1
		# print(demands)

		index = 0
		while index < ficility_count * customer_count:
			line = f.readline().replace('.','').split()
			#print(line)
			for item in line:
				if is_number(item) == False:
					continue
				assignment_costs_matrix[int(index) // int(customer_count)][int(index) % int(customer_count)] = int(item)
				index += 1
		# print(assignment_costs_matrix)

def validate():
	global capacities
	global demands
	cap = 0
	demand = 0
	for value in capacities:
		cap += value
	for value in demands:
		demand += value
	print('capacity:', cap, 'demands:', demand)
	if cap > demand:
		return True
	return False

# 初始化：加载问题实例中的数据
def init(filename):
	global ficilities
	global customers
	global total_cost

	global ficility_count
	global customer_count
	global capacities
	global opening_costs
	global demands
	global assignment_costs_matrix

	load_problem(filename)
	ficilities = []
	customers = []
	total_cost = 0
	for i in range(ficility_count):
		ficilities.append(Ficility(i, capacities[i], opening_costs[i], assignment_costs_matrix[i]))

	for i in range(customer_count):
		customers.append(Customer(i, demands[i]))

# 贪心策略获得初始较优解
def greedy(ficilities, customers):
	cur_ficilities = copy.deepcopy(ficilities)
	cur_customers = copy.deepcopy(customers)
	cur_total_cost = 0
	for customer in cur_customers:
		best_ficility = None
		best_ficility_cost = float('inf')
		#print(customer)
		for ficility in cur_ficilities:
			cost = ficility.get_cost(customer)
			#print('cost:', cost)
			if cost != float('inf') and cost < best_ficility_cost:
				best_ficility_cost = cost
				best_ficility = ficility
		if best_ficility == None:
			print('warning:some customers cannot be assigned to ficility')
		cur_total_cost += best_ficility.assign(customer)
	cur_customers = sort_list(cur_customers)
	return cur_ficilities, cur_customers, cur_total_cost

# 估算当前问题所能达到的最小代价
def eval_min_cost():
	global ficility_count
	global customer_count
	global customers
	global demands
	global assignment_costs_matrix
	global opening_costs
	global capacities
	total_cost = 0
	total_demand = sum(demands)
	total_capacity = sum(capacities)
	total_opening_cost = sum(opening_costs)
	for i in range(customer_count):
		total_cost += min(assignment_costs_matrix[:, i])

	rate = [opening_costs[i] / capacities[i] for i in range(ficility_count)]
	mean_rate = total_opening_cost / total_capacity
	#total_cost += int(min(rate) * total_demand)
	total_cost += int(mean_rate * total_demand)
	return total_cost

def show_solution(ficilities, cost):
	print('solution total cost:%d' % cost)
	for ficility in ficilities:
		print(ficility)

# 测试能否正常加载问题中的数据
def test_load_problems():
	filename = '../instances/p'
	for i in range(71):
		load_problem('../instances/p' + str(i + 1))
		print('../instances/p' + str(i + 1))
		if validate() == False:
			print('failed')

# 爬山法求解
def solve_problem_by_climb(problem_index, max_search_times = 100, round_times = 100):
	global ficilities
	global customers
	st = time.time()
	# 获取初始较优解
	cur_ficilities, cur_customers, cur_cost = greedy(ficilities, customers)
	# 记录算法过程的辅助变量
	epoch = 0
	min_cost = eval_min_cost()
	while epoch < max_search_times:
		best_local_ficilities = None
		best_local_customers = None
		best_local_cost = float('inf')
		# 找出当前解的邻域中round_times个随机解的最优解
		for i in range(round_times):
			new_ficilities, new_customers, cost_diff = get_neighbor(cur_ficilities, cur_customers, cur_cost)
			if cur_cost + cost_diff < best_local_cost:
				best_local_ficilities = new_ficilities
				best_local_customers = new_customers
				best_local_cost = cur_cost + cost_diff
		# 发现更好的解则更新当前解
		if best_local_cost < cur_cost:
			cur_cost = best_local_cost
			cur_ficilities = best_local_ficilities
			cur_customers = best_local_customers
		epoch += 1
		print('当前邻域搜索次数，当前代价，误差:NO.%5d %d %.2f%%' % (epoch, cur_cost, 100 * (cur_cost - min_cost) / min_cost))
	et = time.time()
	t = et -st
	return cur_ficilities, cur_customers, cur_cost, t

# 模拟退火方法求解
def solve_problem_by_SA(problem_index, initial_temp = 100, min_temp = 1, rate = 0.95, round_times = 100):
	global ficilities
	global customers
	st = time.time()
	new_ficilities, new_customers, new_cost = greedy(ficilities, customers)
	min_cost = eval_min_cost()
	new_ficilities, new_customers, new_cost, states = SA(new_ficilities, new_customers, new_cost, initial_temp, min_temp, rate, round_times, min_cost)
	et = time.time()
	t = et - st

	#gen_picture_sa(states)
	return new_ficilities, new_customers, new_cost, t

# 绘制模拟退火的过程图
def gen_picture_sa(states):
	temps = states[0]
	precisions = states[1]
	costs_record = states[2]
	plt.figure()
	plt.title('SA-1')
	plt.plot(temps, color='red', label='temperature')
	plt.xlabel('epoch')
	plt.ylabel('temperature')
	plt.legend()
	ax2 = plt.gca().twinx()
	ax2.plot(precisions, color='blue', label='precision')
	ax2.set_ylabel('cur best precisions')
	plt.legend()

	plt.figure()
	plt.title('SA-2')
	plt.plot(costs_record)
	plt.xlabel('iteration times')
	plt.ylabel('cost')
	plt.show()
	
# 存储多个问题实例的解，最终用于输出csv文件
global results

# 在加载完问题后初始化存放结果的DataFrame
def init_results():
	global results
	
	temp_label = ['' for i in range(ficility_count)]
	temp_label[0] = 'Result'
	temp_label[1] = 'Time(s)'
	label = {'problem': temp_label}
	results = pd.DataFrame(label).transpose()

def append_result(problem_index, ficilities, total_cost, time_elapsed, filename):
	global results
	global ficility_count
	temp_label = ['' for i in range(ficility_count)]
	temp_label[0] = 'Result'
	temp_label[1] = 'Time(s)'
	label = {'problem': temp_label}
	results = pd.DataFrame(label).transpose()

	r = ['' for i in range(len(ficilities))]
	r[0] = total_cost
	r[1] = time_elapsed
	status = []
	assignment = []
	for i in range(len(ficilities)):
		if ficilities[i].total_cost == 0:
			status.append(0)
			assignment.append([])
		else:
			status.append(1)
			temp = [ficilities[i].assignment[j].id for j in range(len(ficilities[i].assignment))]
			assignment.append(temp)
	pname = 'p' + str(problem_index)
	result = {}
	result[pname + '-result'] = r
	result[pname + '-status'] = status
	result[pname + '-assignment'] = assignment
	df = pd.DataFrame(result).transpose()
	results = results.append(df)
	results.to_csv(filename, index = True, header = False, sep=',', mode = 'a+')


def parse_options():
	options = {}
	# default 
	options['isShowFigure'] = False
	options['isRunSA'] = True
	options['willSave'] = True
	options['problem_path'] = '../instances/'
	options['save_path'] = './result/'

def main():
	global ficilities
	global customers
	global results
	#init_results()

	# 使用第一种解法（爬山算法）求解并保存结果到csv文件中
	for i in range(1, 72):
		problem_filename = '../instances/p' + str(i) 
		init(problem_filename)
		#init_results()
		print('load problem p%d finished:' % i)
		min_cost = eval_min_cost()
		print('solving by climb algorithm...')
		new_ficilities, new_customers, new_cost, t = solve_problem_by_climb(i, 50, 20)
		print('cost:', new_cost, 'min cost:', min_cost, 'diff:', int(100 * (new_cost - min_cost) / min_cost), '%')
		print('time elapsed: %.2f seconds' % t)
		append_result(i, new_ficilities, new_cost, t, './results/method_climb.csv')
	#save_to_csv('./results/method_climb.csv')

	# 使用第二种解法（模拟退火算法）求解并保存结果到csv文件中
	for i in range(1, 72):
		problem_filename = '../instances/p' + str(i) 
		init(problem_filename)
		#init_results()
		print('load problem p%d finished:' % i)
		min_cost = eval_min_cost()
		print('solving by SA...')
		new_ficilities, new_customers, new_cost, t = solve_problem_by_SA(i, 100, 5, 0.95, 20)
		print('cost:', new_cost, 'min cost:', min_cost, 'diff:', int(100 * (new_cost - min_cost) / min_cost), '%')
		print('time elapsed: %.2f seconds' % t)
		append_result(i, new_ficilities, new_cost, t, './results/method_SA.csv')
	#save_to_csv('./results/method_SA.csv')
	
if __name__ == '__main__':
	main()

