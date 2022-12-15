def write_xyz(coords, symbols, file_name='test.xyz'):
    num_atoms = len(symbols)
    assert len(coords) == num_atoms    
    with open(file_name, 'w') as file:
        file.write(str(num_atoms) + '\n\n')
        for ida in range(num_atoms):
            file.write(symbols[ida] + ' ' + str(coords[ida][0]) + ' ' + str(coords[ida][1]) + ' ' + str(coords[ida][2]) + '\n')
    return file_name

