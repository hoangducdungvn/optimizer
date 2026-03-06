import matplotlib.pyplot as plt
import numpy as np

def generate_gantt_chart():
    # Dictionary chứa thông tin các công việc: thời gian(duration) và các công việc trước(dependencies)
    tasks = {
        'A': {'desc': 'Lập kế hoạch', 'duration': 4, 'deps': []},
        'B': {'desc': 'Product Backlog', 'duration': 3, 'deps': ['A']},
        'C': {'desc': 'Tài liệu đặc tả Sprint 1', 'duration': 3, 'deps': ['B']},
        'D': {'desc': 'Phân tích & thiết kế Sprint 1', 'duration': 3, 'deps': ['B']},
        'E': {'desc': 'Lập trình (Sprint 1)', 'duration': 9, 'deps': ['B']},
        'F': {'desc': 'Kiểm thử (Sprint 1)', 'duration': 7, 'deps': ['C', 'D', 'E']},
        'G': {'desc': 'Tài liệu đặc tả Sprint 2', 'duration': 3, 'deps': ['F']},
        'H': {'desc': 'Phân tích và thiết kế Sprint 2', 'duration': 3, 'deps': ['F']},
        'I': {'desc': 'Lập trình (Sprint 2)', 'duration': 7, 'deps': ['F']},
        'J': {'desc': 'Kiểm thử (Sprint 2)', 'duration': 7, 'deps': ['G', 'H', 'I']},
    }

    # Tính toán thời gian bắt đầu của từng công việc
    for name, task in tasks.items():
        if not task['deps']:
            task['start'] = 0
        else:
            # Thời gian bắt đầu của 1 task là max thời gian kết thúc của tất cả các task trước nó
            task['start'] = max(tasks[dep]['start'] + tasks[dep]['duration'] for dep in task['deps'])

    # Chuẩn bị dữ liệu vẽ
    fig, ax = plt.subplots(figsize=(12, 7))
    
    y_ticks = []
    y_labels = []
    
    # Màu sắc ngẫu nhiên hoặc theo cmap cho các thanh
    colors = plt.cm.tab10.colors

    # Vẽ từ dưới lên (để task A ở trên cùng)
    for i, (name, task) in enumerate(reversed(list(tasks.items()))):
        y_pos = i * 10
        start = task['start']
        duration = task['duration']
        
        # Vẽ thanh ngang
        facecolor = colors[i % len(colors)]
        ax.broken_barh([(start, duration)], (y_pos + 1, 8), facecolors=facecolor, edgecolor='black')
        
        # Thêm text duration vào giữa thanh
        ax.text(start + duration / 2, y_pos + 5, str(duration), 
                ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        
        # Lưu lại ticks và labels cho trục Y
        y_ticks.append(y_pos + 5)
        y_labels.append(f"({name}) {task['desc']}")

    # Cấu hình trục Y
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # Cấu hình trục X
    max_time = max(task['start'] + task['duration'] for task in tasks.values())
    ax.set_xlim(0, max_time + 1)
    ax.set_xticks(np.arange(0, max_time + 2, 1))

    # Cấu hình Labels và Title
    ax.set_xlabel('Thời gian (ngày)', fontsize=12)
    ax.set_title('Biểu đồ Gantt Lập Kế Hoạch Dự Án', fontsize=16, fontweight='bold', pad=20)

    # Thêm lưới grid
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # Lưu biểu đồ thành file ảnh
    output_filename = 'gantt_chart.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Đã vẽ và lưu biểu đồ vào file: {output_filename}")
    
    # Hiển thị biểu đồ
    plt.show()

if __name__ == '__main__':
    generate_gantt_chart()
