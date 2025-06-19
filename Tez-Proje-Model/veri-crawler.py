import os
from warcio.archiveiterator import ArchiveIterator


def save_warc_pages_as_html(warc_path, output_dir, max_files=22991):
    """ Save each response in a WARC file as a separate HTML file up to a specified number. """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create output directory if it does not exist

    file_count = 0  # Initialize a counter for the number of files saved

    with open(warc_path, 'rb') as stream:
        for i, record in enumerate(ArchiveIterator(stream)):
            if file_count >= max_files:  # Check if the maximum file limit is reached
                print("Reached the maximum limit of files to save.")
                break  # Stop processing if the limit is reached

            if record.rec_type == 'response':
                status_code = record.http_headers.get_statuscode() if hasattr(record.http_headers,
                                                                              'get_statuscode') else \
                record.http_headers.statusline.split()[1]
                if status_code == '200':
                    file_name = f"page_{i + 1}.html"
                    file_path = os.path.join(output_dir, file_name)

                    content = record.content_stream().read()
                    with open(file_path, 'wb') as f:
                        f.write(content)
                    print(f"Saved: {file_path}")
                    file_count += 1  # Increment the file counter after saving a file


# WARC dosya yolu ve çıktı klasörü
warc_file_path = 'CC-MAIN-20240814190250-20240814220250-00368.warc'  # WARC dosyanızın yolu
output_folder = './downladed_html'  # HTML dosyalarının kaydedileceği klasör

# Fonksiyonu çağır
save_warc_pages_as_html(warc_file_path, output_folder)
