import xml.etree.ElementTree as ET

# Load the XML file
tree = ET.parse('tasks.xml')
root = tree.getroot()

# Create dictionaries to store tasks of each stream
stream1_tasks = {}
stream2_tasks = {}

# Iterate over GPU tasks
gpu_tasks = root.find('GPU_TASKS')
for task in gpu_tasks.findall('taskID'):
    task_id = task.get('id')
    name = task.find('name').text
    stream = task.find('stream').text
    next_task = task.find('next').text

    # Store the task in the respective dictionary
    if stream == 'stream1':
        stream1_tasks[task_id] = {'name': name, 'stream': stream, 'next': next_task}
    elif stream == 'stream2':
        stream2_tasks[task_id] = {'name': name, 'stream': stream, 'next': next_task}

# Update "next" of each task in stream1
stream1_task_ids = list(stream1_tasks.keys())
for i in range(len(stream1_task_ids) - 1):
    current_task_id = stream1_task_ids[i]
    next_task_id = stream1_task_ids[i + 1]
    stream1_tasks[current_task_id]['next'] = stream1_tasks[next_task_id]['name']

# Update "next" of each task in stream2
stream2_task_ids = list(stream2_tasks.keys())
for i in range(len(stream2_task_ids) - 1):
    current_task_id = stream2_task_ids[i]
    next_task_id = stream2_task_ids[i + 1]
    stream2_tasks[current_task_id]['next'] = stream2_tasks[next_task_id]['name']

# Update the corresponding GPU task's "next" value
for task_id, task in stream1_tasks.items():
    gpu_task = gpu_tasks.find(f"taskID[@id='{task_id}']")
    next_task_element = gpu_task.find('next')
    next_task_element.text = task['next']

for task_id, task in stream2_tasks.items():
    gpu_task = gpu_tasks.find(f"taskID[@id='{task_id}']")
    next_task_element = gpu_task.find('next')
    next_task_element.text = task['next']

# Write the updated XML back to the file
tree.write('updated_tasks.xml')
