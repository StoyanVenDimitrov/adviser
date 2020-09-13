x = []
ep = []
val = []

def average(value, index):
    # use to write training loss for averaging
    try:
        x[index] += value
    except IndexError:
        x.append(value)


def write_summ(summary_writer, iterations):
    iterations += 1
    train_call_count = 0
    for value in x:
        train_call_count += 1
        value = value/iterations
        summary_writer.add_scalar('avg/loss', value, train_call_count)

def average_val(value, index):
    # use to write val episode reward for averaging
    try:
        val[index] += value
    except IndexError:
        val.append(value)


def write_summ_val(summary_writer, iterations):
    iterations += 1
    train_call_count = 0
    for value in val:
        value = value/iterations
        summary_writer.add_scalar('eval/avg_reward', value, train_call_count)
        train_call_count += 1


def reward_average(value, index):
    # use to write train episode reward for averaging
    try:
        ep[index] += value
    except IndexError:
        ep.append(value)


def write_summ_reward(summary_writer, iterations):
    iterations += 1
    train_call_count = 0
    for value in ep:
        train_call_count += 1
        value = value/iterations
        summary_writer.add_scalar('avg/ep_reward', value, train_call_count)

def shutdown_conf():
    x = []
    ep = []
    val = []
    print('shuting down')
