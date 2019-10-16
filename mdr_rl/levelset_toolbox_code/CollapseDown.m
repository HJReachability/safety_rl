function  [new_data, new_g]  = CollapseDown(data, g, choice,fix)
  orig_data_size = size(data);

  old_dim=g.dim;

%   for k=1:length(fix)
%       mid_index(k)=floor(median(1:orig_data_size(fix(k))));
%   end
%
  label1=['new_data=data('];

  for k=1:old_dim
      if sum(k==choice)>0
          label1=[label1 ':,'];
      else
          label1=[label1 [num2str(floor(median(1:orig_data_size(k)))),',']];
      end
  end

  label1=label1(1:end-1);
  label1=[label1 ');'];


  label2=['new_data=reshape(new_data,'];

  for k=1:length(choice)

      label2=[label2 [num2str(orig_data_size(choice(k))),',']];
  end

  label2=label2(1:end-1);
  label2=[label2 ');'];

  new_indices=choice;
  eval(label1)
  eval(label2)


%   if choice == 1
%     new_indices = [1,2,3];
%     mid_index_1 = floor(median(1:orig_data_size(4)));
%     new_data = data(:,:,:,mid_index_1);
%     new_data = reshape(new_data, orig_data_size(1),orig_data_size(3),orig_data_size(5));
%   elseif choice == 2
%     new_indices = [1,2,4];
%     mid_index_1 = floor(median(1:orig_data_size(3)));
%     new_data = data(:,:,mid_index_1,:);
%     new_data = reshape(new_data, orig_data_size(2),orig_data_size(4),orig_data_size(5));
%   else
%     new_indices = [1,2,5];
%     mid_index_1 = floor(median(1:orig_data_size(3)));
%     mid_index_2 = floor(median(1:orig_data_size(4)));
%     new_data = data(:,:,mid_index_1,mid_index_2,:);
%     new_data = reshape(new_data, orig_data_size(1),orig_data_size(2),orig_data_size(5));
%   end



  new_g.dim = length(new_indices);
  new_g.min = g.min(new_indices);
  new_g.max = g.max(new_indices);
  new_g.bdry = g.bdry(new_indices);
  new_g.N = g.N(new_indices);
  new_g.dx = g.dx(new_indices);
  new_g.vs = g.vs(new_indices);
  new_g.xs = g.xs(new_indices);

  label3=['new_g.xs{i}=new_g.xs{i}('];

  for k=1:old_dim
      if sum(k==choice)>0
          label3=[label3 ':,'];
      else
          label3=[label3 [num2str(floor(median(1:orig_data_size(k)))),',']];
      end
  end

  label3=label3(1:end-1);
  label3=[label3 ');'];


  label4=['new_g.xs{i}=reshape(new_g.xs{i},'];

  for k=1:length(choice)

      label4=[label4 [num2str(orig_data_size(choice(k))),',']];
  end

  label4=label4(1:end-1);
  label4=[label4 ');'];




  for i=1:new_g.dim
      eval(label3)
      eval(label4)
  end

%   for i=1:new_g.dim
%     if choice == 1
%       new_g.xs{i} = new_g.xs{i}(:,mid_index_1,:,mid_index_2,:);
%       new_g.xs{i} = reshape(new_g.xs{i}, orig_data_size(1),orig_data_size(3),orig_data_size(5));
%     elseif choice == 2
%       new_g.xs{i} = new_g.xs{i}(mid_index_1,:,mid_index_2,:,:);
%       new_g.xs{i} = reshape(new_g.xs{i}, orig_data_size(2),orig_data_size(4),orig_data_size(5));
%     else
%       new_g.xs{i} = new_g.xs{i}(:,:,mid_index_1,mid_index_2,:);
%       new_g.xs{i} = reshape(new_g.xs{i}, orig_data_size(1),orig_data_size(2),orig_data_size(5));
%     end
%   end
  new_g.bdryData = g.bdryData(new_indices);
  new_g.shape = g.shape(new_indices);
  new_g.axis = g.axis;

return;