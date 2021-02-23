import os
import shutil
import sys
from tempfile import mkdtemp
import errno
import subprocess

# source directory to access files from
source = os.path.dirname(os.path.realpath(__file__))+"/test"

# destination directory to save txt file to.
destination = os.path.dirname(os.path.realpath(__file__))+"/test"

# trying to import specific modules

try:
    from pdf2image import convert_from_path, convert_from_bytes
except ImportError:
    print('Error: You need to install the "pdf2image" package. Type the following:')
    print('pip install pdf2image')
    exit()

try:
    from PIL import Image
except ImportError:
    print('Error: You need to install the "Image" package. Type the following:')
    print('pip install Image')

try:
    import pytesseract
except ImportError:
    print('Error: You need to install the "pytesseract" package. Type the following:')
    print('pip install pytesseract')
    exit()


# specific modules import ended

def update_movement(movement):
    """
    tracks the progress of the conversion.
    :param movement: Takes the floating values of progress to reflect it in the farm of progress bar
    :return: None
    """

    # Modify this to change the length of the movement bar
    progress_bar_length = 50
    response = ""

    if isinstance(movement, int):
        movement = float(movement)

    if not isinstance(movement, float):
        movement = 0
        response = "[X]Error: Movement must be in float.\n"

    if movement < 0:
        movement = 0
        response = "Progress is stopped.\n"
    if movement >= 1:
        movement = 1
        response = "\n"

    block = int(round(progress_bar_length * movement))
    text = "\rPercent: [{0}] {1}% {2}".format("*" * block + "->" * (progress_bar_length - block), movement * 100,
                                              response)
    sys.stdout.write(text)
    sys.stdout.flush()


def run(args):
    """
    run a subprocess and put the stdout and stderr on the pipe object.
    :param args: sequence of program that needs to be executed.
    :return: stdout and stderr references
    """
    try:

        pipe = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    except OSError as e:

        if e.errno == errno.ENOENT:
            # In case of file not found
            raise Exception(' '.join(args), 127, '', '')

    stdout, stderr = pipe.communicate()

    # if pipe is not responding.
    if pipe.returncode != 0:
        raise Exception(' '.join(args), pipe.returncode, stdout, stderr)

    return stdout, stderr


def get_text_tesseract(filename):
    """
    Method to uslilize the tesseract to extract the content of the
    pdf/image file.
    :param filename: file from where the text is to be extracted.
    :return:the content of the file in text form (string in a list)
    """

    # creating a temporary directory
    temp_dir = mkdtemp()
    base = os.path.join(temp_dir, 'conv')
    contents = []
    try:
        stdout, _ = run(['pdftoppm', filename, base])

        for page in sorted(os.listdir(temp_dir)):
            page_path = os.path.join(temp_dir, page)
            page_content = pytesseract.image_to_string(Image.open(page_path))
            contents.append(page_content)

        return ''.join(contents)

    finally:

        shutil.rmtree(temp_dir)


def recursive_convert(source, destination, count):
    """
    The method to conver the pdf recursively that are present in the directory.
    :param source: the path to the files directory
    :param destination: the path to the directory where you want to save the converted files
    :param count: the number of files that are converted
    :return: the number of files that are converted
    """
    count_pdf_files = 0

    for dirpath, dirnames, files in os.walk(source):

        for file in files:

            if file.lower().endswith('.pdf'):
                count_pdf_files += 1

    ''' Helper function for looping through files recursively '''
    for dirpath, dirnames, files in os.walk(source):

        for name in files:

            filename, file_extension = os.path.splitext(name)

            if file_extension.lower() != '.pdf':
                continue

            relative_directory = os.path.relpath(dirpath, source)
            source_path = os.path.join(dirpath, name)
            output_directory = os.path.join(destination, relative_directory)
            output_filename = os.path.join(output_directory, filename + '.txt')

            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            count = convert(source_path, output_filename, count, count_pdf_files)

    return count


def convert(sourcefile, destination_file, count, count_pdf_files):
    """
    The method to convert the files and store them as txt files with same name.
    :param sourcefile: source pdf file address
    :param destination_file: destination to save the converted text file(utf-8 encoding)
    :param count:
    :param count_pdf_files:
    :return: number of files that are converted.
    """

    text = get_text_tesseract(sourcefile)

    with open(destination_file, 'w', encoding='utf-8') as f_out:
        f_out.write(text)

    print('Converted ' + source)
    count += 1
    update_movement(count / count_pdf_files)

    return count


count = 0

if os.path.exists(source):
    if os.path.isdir(source):
        count = recursive_convert(source, destination, count)
    elif os.path.isfile(source):
        filepath, fullfile = os.path.split(source)
        filename, file_extension = os.path.splitext(fullfile)
        if file_extension.lower() == '.pdf':
            count = convert(source, os.path.join(destination, filename + '.txt'), count, 1)
    plural = 's'
    if count == 1:
        plural = ''
    print(str(count) + ' file' + plural + ' converted to text.')
else:
    print('The path ' + source + 'is not valid')
