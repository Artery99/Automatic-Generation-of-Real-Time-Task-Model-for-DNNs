import re
import xml.etree.ElementTree as ET

class GPUElement:
    def __init__(self, kernel_name, stream=None, data=None, next=None, size=0):
        self.kernel_name = kernel_name
        self.stream = stream
        self.data = data
        self.size = size
        self.next = next
        self.Type = None  # Initialize Type attribute

    def __str__(self):
        if self.stream and self.data:
            return f'<GPU: kernel_name: {self.kernel_name} | stream: {self.stream} | data: {self.data} | next: {self.next} | size: {self.size} | Type: {self.Type}>'
        return f'<GPU: kernel_name: {self.kernel_name} | Type: {self.Type}>'

class CPUElement:
    def __init__(self, kernel_name, gpu_kernel_name=None, next=None, node=None):
        self.kernel_name = kernel_name
        self.created_by = gpu_kernel_name
        self.next = next
        self.node = node

    def __str__(self):
        if not self.created_by:
            return f'<CPU: kernel_name: {self.kernel_name} | next: {self.next} | node: {self.node}>'
        return f'<CPU: kernel_name: {self.kernel_name} | created by GPU kernel {self.created_by} | next: {self.next} | node: {self.node}>'

# a hashmap from dim3 type variabled name to passed arguments
vars_map = {}
# list of only gpu instances
gpu_instances = []
# list of only cpu instances
cpu_instances = []
# defined with __global__
defined_gpu_kernels = []
# defined with host
defined_hosts_list = []

GPU_list = []
CPU_list = []
Streams = []
Data = []
Order = []
Type = []
Node = []
Associated_CPUs = []

with open("yolo-v3_tiny-2Streams.cu", "r") as f:
    for line in f.readlines():

        match = re.match("(__global__|host) void ([a-zA-Z_0-9]+)",line)
        if match:
            x,y = match.group(1), match.group(2)
            if x == '__global__':
                GPU_list = GPU_list + [y]
            if x == 'host':
                CPU_list = CPU_list + [y]

with open("yolo-v3_tiny-2Streams.cu", "r") as f:
    for line in f.readlines():
        match = re.search("0, (.*?)>>>", line)
        if match:
            stream = match.group(1)
            Streams.append(stream)

with open("yolo-v3_tiny-2Streams.cu", "r") as f:
    for line in f.readlines():
        matches = re.findall(">>>(.*?)(?=\))|(" + "|".join(CPU_list) + r")\((.*?)\);", line)
        if matches:
            params_list = []
            for match in matches:
                if match[0]:
                    params_list.append(match[0].strip()[1:])
                elif match[2]:
                    params_list.append(match[2].strip())
            Data.append(",".join(params_list))

size = 1
with open("yolo-v3_tiny-2Streams.cu", "r") as f:
    for line in f.readlines():
        match = re.search("dim3 grid\((.*?)\)", line)
        if match:
            numbers = match.group(1).split(',')
            for num in numbers:
                size *= (int(num.strip()))

with open("yolo-v3_tiny-2Streams.cu", "r") as f:
    start_scanning = False
    for line in f.readlines():
        if "yolov3Tiny" in line:
            start_scanning = True
        if start_scanning:
            match = re.search("|".join(GPU_list + CPU_list), line)
            if match:
                Order.append(match.group(0))
                break

with open("yolo-v3_tiny-2Streams.cu", "r") as f:
    start_scanning = False
    prev_match = None
    prev_gpu_index = None
    for line in f.readlines():
        if "yolov3Tiny" in line:
            start_scanning = True
        if start_scanning:
            matches = re.findall("|".join(GPU_list + CPU_list), line)
            for match in matches:
                if match in GPU_list:
                    gpu_index = GPU_list.index(match)
                    if prev_match in CPU_list:
                        Order.append(match)
                    else:
                        if prev_gpu_index is not None and prev_gpu_index < gpu_index and Streams[prev_gpu_index] == Streams[gpu_index]:
                            Order.append(match)
                    prev_gpu_index = gpu_index
                elif match in CPU_list:
                    Order.append(match)
                prev_match = match
        if "return 0;" in line:
            break

Type = GPU_list + CPU_list
Node = CPU_list
Associated_CPUs = [gpu + "'" for gpu in GPU_list]



with open('yolo-v3_tiny-2Streams.cu', 'r') as f:
    for line in f.readlines():
        matched_global = re.match('__global__ void (.*)\(.*', line)
        matched_main = re.match('(.*)<<<(.*), (.*), (.*), (.*)>>>\((.*)\)', line)
        matched_dim3 = re.match('   dim3 (.*)\((.*)\)', line)
        matched_host = re.match('host void (.*)\(', line)

        if matched_global:
            defined_gpu_kernels.append(matched_global.group(1))
        if matched_dim3:
            var_name, args = matched_dim3.group(1), matched_dim3.group(2)
            vars_map[var_name] = args

        if matched_main:
            kernel_name, var_name_1, var_name_2, _, stream_name, data \
                = (matched_main.group(i) for i in range(1, 7))
            gpu_instances.append(
                GPUElement(
                    kernel_name=kernel_name.strip(),
                    stream=stream_name,
                    data=data.split(', '),
                    size=size
                )
            )
            cpu_instances.append(
                CPUElement(
                    kernel_name='custom_cpu_name',
                    gpu_kernel_name=kernel_name
                )
            )

            # TODO: create a CPU for each GPU
            # TODO: handle next, what is next?
        if matched_host:
            defined_hosts_list.append(matched_host.group(1))

gpu_instances_names = list(map(lambda x: x.kernel_name, gpu_instances))
for kernel_name in defined_gpu_kernels:
    # loop on the GPU element with no stream
    if kernel_name not in gpu_instances_names:
        gpu_instances.append(
            GPUElement(kernel_name=kernel_name)
        )
        # TODO: should you create a CPU element for this one?
del gpu_instances_names

for kernel_name in defined_hosts_list:
    cpu_instances.append(
        CPUElement(kernel_name=kernel_name)
    )

for i in range(len(gpu_instances)):
    if i < len(gpu_instances) - 1:
        gpu_instances[i].next = gpu_instances[i + 1].kernel_name
    else:
        gpu_instances[i].next = None

for i in range(len(cpu_instances)):
    if i < len(cpu_instances) - 1:
        cpu_instances[i].next = cpu_instances[i + 1].kernel_name
    else:
        cpu_instances[i].next = None

# Initialize the Type attribute for each GPU element
for gpu, type_value in zip(gpu_instances, Type):
    gpu.Type = type_value

for i in range(len(cpu_instances)):
    if i < len(cpu_instances) - 1:
        cpu_instances[i].next = Order[i+1] if i+1 < len(Order) else None
    else:
        cpu_instances[i].next = None
    cpu_instances[i].node = Node[i] if i < len(Node) else None


for i in range(len(cpu_instances)):
    if cpu_instances[i].kernel_name == "custom_cpu_name":
        cpu_instances[i].node = None
    else:
        cpu_instances[i].node = Node[i] if i < len(Node) else None

for e in [*gpu_instances, *cpu_instances]:
    print(e)

# Define the indentation function
def indent(elem, level=0):
    # Indentation string
    indent_str = "    "
    # Newline string
    newline_str = "\n" + indent_str * level

    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = newline_str + indent_str
        if not elem.tail or not elem.tail.strip():
            elem.tail = newline_str
        for sub_elem in elem:
            indent(sub_elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = newline_str
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = newline_str

# Create GPU tasks XML
gpu_tasks = ET.Element('GPU_TASKS')
for i, gpu in enumerate(gpu_instances):
    task_id = i + 1
    task_element = ET.SubElement(gpu_tasks, 'taskID', id=str(task_id))
    ET.SubElement(task_element, 'name').text = gpu.kernel_name
    ET.SubElement(task_element, 'stream').text = str(gpu.stream)
    ET.SubElement(task_element, 'data').text = str(gpu.data)
    ET.SubElement(task_element, 'next').text = str(gpu.next)
    ET.SubElement(task_element, 'size').text = str(gpu.size)
    ET.SubElement(task_element, 'Type').text = str(gpu.Type)

# Create CPU tasks XML
cpu_tasks = ET.Element('CPU_TASKS')
for i, cpu in enumerate(cpu_instances):
    task_id = i + 1
    task_element = ET.SubElement(cpu_tasks, 'taskID', id=str(task_id))
    ET.SubElement(task_element, 'name').text = cpu.kernel_name
    ET.SubElement(task_element, 'created_by').text = str(cpu.created_by)
    ET.SubElement(task_element, 'next').text = str(cpu.next)
    ET.SubElement(task_element, 'node').text = str(cpu.node)

# Create the root element for the XML
root = ET.Element('TASKS')
root.append(gpu_tasks)
root.append(cpu_tasks)

# Indent the XML elements
indent(root)

# Create the XML tree
tree = ET.ElementTree(root)

# Write the XML to a file
output_file = 'tasks.xml'
tree.write(output_file)

# Read the generated XML file and print its contents
with open(output_file, 'r') as f:
    xml_content = f.read()

print(f"XML file '{output_file}' has been generated successfully.")
print("\nXML Content:")
print(xml_content)