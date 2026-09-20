function pearl_apce_and_edge_f1(night_IR_folder, generated_folder, output_folder, generated_suffix)
%PEARL_APCE_AND_EDGE_F1 Evaluate TIR-to-RGB structural consistency.
%
% APCE preserves PearlGAN's official APCE_eval/batch_eval_CE_FLIR_single.m:
% MATLAB Canny; 99 high thresholds (0.01:0.01:0.99); low=0.5*high; and
% averaging only images with non-empty TIR edges. Changes are limited to
% function parameters and output filename mapping for FLIR v2/CycleGAN.

    if nargin < 4
        error('Usage: pearl_apce_and_edge_f1(night_IR_folder, generated_folder, output_folder, generated_suffix)');
    end
    if ~exist(night_IR_folder, 'dir')
        error('TIR directory does not exist: %s', night_IR_folder);
    end
    if ~exist(generated_folder, 'dir')
        error('Generated-image directory does not exist: %s', generated_folder);
    end
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    extensions = {'*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff'};
    dir_file = [];
    for e = 1:numel(extensions)
        dir_file = [dir_file; dir(fullfile(night_IR_folder, extensions{e}))]; %#ok<AGROW>
    end
    if isempty(dir_file)
        error('No supported TIR images found in: %s', night_IR_folder);
    end
    [~, order] = sort({dir_file.name});
    dir_file = dir_file(order);
    day_file_names = {dir_file.name};

    [~, method_name] = fileparts(generated_folder);
    apce_path = fullfile(output_folder, [method_name, '_APCE.txt']);
    prf_path = fullfile(output_folder, [method_name, '_edge_PRF.csv']);
    fid_apce = fopen(apce_path, 'w');
    fid_prf = fopen(prf_path, 'w');
    if fid_apce == -1 || fid_prf == -1
        error('Could not open output files in: %s', output_folder);
    end
    cleanup = onCleanup(@() close_files(fid_apce, fid_prf)); %#ok<NASGU>
    fprintf(fid_prf, 'high_threshold,precision,recall,f1,valid_images\n');

    AP_array = zeros(1, 99);
    F1_array = zeros(1, 99);
    for j = 1:99
        high_th = j * 0.01;
        low_th = high_th * 0.5;
        precise_ratio = 0.0;
        precision_ratio = 0.0;
        f1_ratio = 0.0;
        cnt = 0;

        for i = 1:length(day_file_names)
            img_name = day_file_names{1, i};
            [~, stem, ~] = fileparts(img_name);
            if isempty(generated_suffix)
                vis_img_file = fullfile(generated_folder, img_name);
            else
                vis_img_file = fullfile(generated_folder, [stem, generated_suffix]);
            end
            if ~exist(vis_img_file, 'file')
                error('Missing generated image for %s. Expected: %s', img_name, vis_img_file);
            end

            IR_img = imread(fullfile(night_IR_folder, img_name));
            vis_img = imread(vis_img_file);
            % These two calls and APCE recall expression match PearlGAN exactly.
            IR_edge = edge(IR_img, 'canny', [low_th, high_th]);
            vis_edge = edge(rgb2gray(vis_img), 'canny', [low_th, high_th]);
            ir_count = sum(sum(double(IR_edge)));
            vis_count = sum(sum(double(vis_edge)));
            if ir_count > 0
                cnt = cnt + 1;
                overlap = sum(sum(double(vis_edge) .* double(IR_edge)));
                temp_precise_ratio = overlap / ir_count;
                precise_ratio = precise_ratio + temp_precise_ratio;
                if vis_count > 0
                    precision_ratio = precision_ratio + overlap / vis_count;
                end
                f1_ratio = f1_ratio + 2 * overlap / (ir_count + vis_count);
            end
        end
        if cnt == 0
            error('No non-empty TIR edge maps at threshold %.2f.', high_th);
        end
        final_recall = precise_ratio / cnt;
        final_precision = precision_ratio / cnt;
        final_f1 = f1_ratio / cnt;
        % Preserve the official APCE file's two-column format.
        fprintf(fid_apce, '%.2f %.6f\n', high_th, final_recall);
        fprintf(fid_prf, '%.2f,%.6f,%.6f,%.6f,%d\n', high_th, final_precision, final_recall, final_f1, cnt);
        AP_array(1, j) = final_recall;
        F1_array(1, j) = final_f1;
    end
    summary_path = fullfile(output_folder, [method_name, '_edge_summary.txt']);
    fid_summary = fopen(summary_path, 'w');
    fprintf(fid_summary, 'APCE %.6f\n', sum(AP_array) / 99);
    fprintf(fid_summary, 'EdgeF1 %.6f\n', sum(F1_array) / 99);
    fclose(fid_summary);
    fprintf('APCE = %.6f\n', sum(AP_array) / 99);
    fprintf('EdgeF1 = %.6f\n', sum(F1_array) / 99);
end

function close_files(fid_apce, fid_prf)
    if fid_apce ~= -1, fclose(fid_apce); end
    if fid_prf ~= -1, fclose(fid_prf); end
end
