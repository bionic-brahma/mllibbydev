import subprocess
import os.path
import sys

class CompressPDF:
    def __init__(self, compress_level=0, show_info=False):
        self.compress_level = compress_level

        self.quality = {
            0: '/default',
            1: '/prepress',
            2: '/printer',
            3: '/ebook',
            4: '/screen'
        }

        self.show_compress_info = show_info

    def compress(self, file=None, new_file=None):
        """
        Function to compress PDF via Ghostscript command line interface

        :param file: old file that needs to be compressed
        :param new_file: new file that is commpressed
        :return: True or False, to do a cleanup when needed
        """
        global initial_size

        try:

            if not os.path.isfile(file):

                print("Error: invalid path for input PDF file")
                sys.exit(1)

            # check if file is a PDF by extension
            filename, file_extension = os.path.splitext(file)

            if file_extension != '.pdf':

                raise Exception("Error: input file is not a PDF")

            if self.show_compress_info:

                initial_size = os.path.getsize(file)

            subprocess.call(['gs', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.4',
                             '-dPDFSETTINGS={}'.format(self.quality[self.compress_level]),
                             '-dNOPAUSE', '-dQUIET', '-dBATCH',
                             '-sOutputFile={}'.format(new_file),
                             file]
                            )

            if self.show_compress_info:

                final_size = os.path.getsize(new_file)
                ratio = 1 - (final_size / initial_size)
                print("Compression by {0:.0%}.".format(ratio))
                print("Final file size is {0:.1f}MB".format(final_size / 1000000))

            return True

        except Exception as error:

            print('Caught this error: ' + repr(error))

        except subprocess.CalledProcessError as e:

            print("Unexpected error:".format(e.output))

            return False


if __name__ == '__main__':

    start_folder = "test/"
    compress = 2
    p = CompressPDF(compress, show_info=True)

    compress_folder = os.path.join(start_folder, "test_out/")

    for filename in os.listdir(start_folder):

        my_name, file_extension = os.path.splitext(filename)

        if file_extension == '.pdf':

            file = os.path.join(start_folder, filename)

            new_file = os.path.join(compress_folder, filename)

            if p.compress(file, new_file):

                print("{} Compression done.".format(filename))

            else:

                print("Format Error")
