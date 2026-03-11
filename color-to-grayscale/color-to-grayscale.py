def color_to_grayscale(image):
    gray = []
    for row in image:
        gray_row = []
        for r, g, b in row:
            gray_row.append(0.299*r + 0.587*g + 0.114*b)
        gray.append(gray_row)
    return gray
            