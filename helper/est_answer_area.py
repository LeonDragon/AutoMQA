import numpy as np 

def infer_answer_area_average_size(bubble_coords):
    """Infers the answer area on an OMR sheet by analyzing bubble coordinates.
    This function estimates the bounding box that encompasses the answer area 
    by calculating the average bubble size and filtering out outliers.
    Args:
        bubble_coords (list): A list of tuples, where each tuple 
                              represents the coordinates of a detected bubble 
                              in the format (x, y, w, h), where:
                                - x: x-coordinate of the top-left corner
                                - y: y-coordinate of the top-left corner
                                - w: width of the bubble
                                - h: height of the bubble
    Returns:
        tuple: A tuple containing the coordinates of the inferred answer area 
               in the format (x_min, y_min, x_max, y_max). 
    """        

    if not bubble_coords:
        return None, None, None, None  # Return None for all values

    # Calculate average bubble size
    avg_width = sum(w for _, _, w, _ in bubble_coords) / len(bubble_coords)
    avg_height = sum(h for _, _, _, h in bubble_coords) / len(bubble_coords)

    # Filter bubbles based on size similarity
    filtered_coords = []
    for x, y, w, h in bubble_coords:
        if abs(w - avg_width) < 0.2 * avg_width and abs(h - avg_height) < 0.2 * avg_height:
            filtered_coords.append((x, y, w, h))

    if filtered_coords:
        x_min = min(x for x, _, _, _ in filtered_coords)
        y_min = min(y for _, y, _, _ in filtered_coords)
        x_max = max(x + w for x, _, w, _ in filtered_coords)
        y_max = max(y + h for _, y, _, h in filtered_coords)

        return x_min, y_min, x_max, y_max  # Return the coordinates

    else:
        return None, None, None, None  # Return None for all values

def infer_answer_area_row_col(bubble_coords):
    """Infers the answer area using row/column analysis."""

    if not bubble_coords:
        return None

    # Sort by y-coordinate (rows)
    bubble_coords.sort(key=lambda coord: coord[1])  # Sort by y

    rows = []
    current_row = []
    last_y = bubble_coords[0][1]
    row_threshold = 20  # Adjust this threshold based on bubble spacing

    for x, y, w, h in bubble_coords:
        if abs(y - last_y) > row_threshold:
            # Start a new row
            rows.append(current_row)
            current_row = []
        current_row.append((x, y, w, h))
        last_y = y
    rows.append(current_row)  # Add the last row

    # Sort within each row by x-coordinate (columns)
    for row in rows:
        row.sort(key=lambda coord: coord[0])  # Sort by x

    # (Optional) Analyze distances between bubbles to refine row/column detection
    # ...

    # Define answer area using outermost bubbles
    x_min = min(row[0][0] for row in rows)
    y_min = rows[0][0][1]
    x_max = max(row[-1][0] + row[-1][2] for row in rows)
    y_max = rows[-1][0][1] + rows[-1][0][3]

    # Add some padding
    padding = 15
    x_min -= padding
    y_min -= padding
    x_max += padding
    y_max += padding

    return x_min, y_min, x_max, y_max

def infer_answer_area_expanding_box(bubble_coords):
    """Infers the answer area using an expanding bounding box."""

    if not bubble_coords:
        return None

    # Select initial bubbles
    bubble_coords.sort(key=lambda coord: (coord[0], coord[1]))  # Sort by x, then y
    top_left = bubble_coords[0]
    bottom_right = bubble_coords[-1]

    # Initial bounding box
    x_min, y_min = top_left[0], top_left[1]
    x_max, y_max = bottom_right[0] + bottom_right[2], bottom_right[1] + bottom_right[3]

    expansion_increment = 10  # Adjust this increment
    max_iterations = 50  # Adjust this limit
    previous_count = 0
    stable_count = 0
    stability_threshold = 3  # Adjust this threshold

    for _ in range(max_iterations):
        bubble_count = 0
        for x, y, w, h in bubble_coords:
            if x_min <= x and y_min <= y and x + w <= x_max and y + h <= y_max:
                bubble_count += 1

        if bubble_count == previous_count:
            stable_count += 1
        else:
            stable_count = 0
        previous_count = bubble_count

        if stable_count >= stability_threshold:
            break

        # Expand the box
        x_min -= expansion_increment
        y_min -= expansion_increment
        x_max += expansion_increment
        y_max += expansion_increment

    return x_min, y_min, x_max, y_max

def remove_outliers(data, threshold=1.5):
    """Removes outliers from a list of data using the IQR method.

    Args:
      data: A list of numerical data.
      threshold: The IQR multiplier for determining outliers.

    Returns:
      A new list with the outliers removed.
    """

    data = np.array(data)  # Convert to NumPy array for easier calculations
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return data[(data >= lower_bound) & (data <= upper_bound)].tolist()

def infer_answer_area_grid(bubble_coords):
    """Infers the answer area on an OMR sheet with a grid layout by analyzing 
    bubble coordinates.

    This function estimates the bounding box of the answer area by identifying 
    the grid structure based on similar x and y coordinates of the bubbles.
    It also attempts to eliminate outliers on each side of the grid using 
    the IQR method.

    Args:
        bubble_coords (list): A list of tuples, where each tuple 
                              represents the coordinates of a detected bubble 
                              in the format (x, y, w, h), where:
                                - x: x-coordinate of the top-left corner
                                - y: y-coordinate of the top-left corner
                                - w: width of the bubble
                                - h: height of the bubble

    Returns:
        tuple: A tuple containing the coordinates of the inferred answer area 
               in the format (x_min, y_min, x_max, y_max).
               Returns (None, None, None, None) if no answer area can be inferred.
    """

    if not bubble_coords:
        return None, None, None, None

    # Sort bubbles by x-coordinate
    bubble_coords.sort(key=lambda coord: coord[0])

    print("===============")
    print("bubble EST:", bubble_coords)

    x_left_values = []
    x_right_values = []

    # --- Identify Left and Right Boundaries ---
    x_threshold = 10  # Adjust this threshold for x-coordinate similarity
    current_x = bubble_coords[0][0]
    x_left_values.append(current_x)
    for x, _, w, _ in bubble_coords:
        if abs(x - current_x) > x_threshold:
            # Significant change in x, likely a new column
            x_right_values.append(current_x + w)  # Right edge of previous column
            x_left_values.append(x)  # Left edge of new column
            current_x = x

    x_right_values.append(bubble_coords[-1][0] + bubble_coords[-1][2])  # Last column

    # --- Eliminate Outliers (Left and Right) ---
    print("x_left_values before outlier removal:", x_left_values)
    print("x_right_values before outlier removal:", x_right_values)
    x_left_values = remove_outliers(x_left_values)
    x_right_values = remove_outliers(x_right_values)

    # --- Identify Top and Bottom Boundaries ---
    bubble_coords.sort(key=lambda coord: coord[1])  # Sort by y
    y_threshold = 10  # Adjust threshold for y-coordinate similarity
    current_y = bubble_coords[0][1]
    y_top_values = [current_y]
    y_bottom_values = []  # Initialize y_bottom_values here
    for _, y, _, h in bubble_coords:
        if abs(y - current_y) > y_threshold:
            y_bottom_values.append(current_y + h)
            y_top_values.append(y)
            current_y = y
    y_bottom_values.append(bubble_coords[-1][1] + bubble_coords[-1][3])

    # --- Eliminate Outliers (Top and Bottom) ---
    print("y_top_values before outlier removal:", y_top_values)
    print("y_bottom_values before outlier removal:", y_bottom_values)
    y_top_values = remove_outliers(y_top_values)
    y_bottom_values = remove_outliers(y_bottom_values)

    if x_left_values and x_right_values and y_top_values and y_bottom_values:
        x_min = min(x_left_values)
        y_min = min(y_top_values)
        x_max = max(x_right_values)
        y_max = max(y_bottom_values)
        return x_min, y_min, x_max, y_max
    else:
        return None, None, None, None