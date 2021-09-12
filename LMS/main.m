%Name: Automatically reverb removing based on LMS
%Author: Jiawei Zheng
%email: hu20327@bristol.ac.uk
%audioset: from Mozzila Common Voice and use Adoble Audition to add reverb
%- [Mozilla Common Voice](https://voice.mozilla.org/)
%- [Adoble Audition](https://www.adobe.com/cn/products/audition.html)
number_order= 50; %the min number of orders
[reverb_audio,sr]=mp3read( 'common_voice_zh-SP_18524189_reverb.mp3'); %reverb audio
reverb_audio=reverb_audio(:);
 
[clean_audio,sr] = mp3read( 'common_voice_zh-SP_18524189.mp3');%orignal audio
clean_audio=clean_audio(:);
total_length= size(clean_audio, 1); %stride
N=length(reverb_audio);
k= 0: 691246;
 
%begin the algorithm
WH_filter= zeros(number_order, 1); %initialize the Windrow-Hoff filters
WH_filter_1= zeros(number_order, 1); %initialize the Windrow-Hoff filters
error= zeros(N, 1); %initialize the error array
final_error= zeros(N, 1);
Loop= 15000;
 
for n=number_order:N
input=reverb_audio(n:- 1:n-number_order+ 1); %enter the reverb input
%% Improved LMS
output(n)=WH_filter'*input; %compute output
r(n)=input'*input; %autocorrelation
e(n)=clean_audio(n)-output(n); %error
srr(n)= 10* log10(clean_audio(n)/(reverb_audio(n)-clean_audio(n)));
srr1(n)= 10* log10(clean_audio(n)/(output(n)-clean_audio(n)));

correct_factor= 1e-10; %correct fact to avoid the output from being divided by zeor
if n< 2000
learning_rate= 0.32;
else
learning_rate= 0.15;
end
WH_filter= 0.8*WH_filter+learning_rate*input*e(n)/(r(n)+correct_factor); %iteration of Improved LMS
error(n)=error(n)+e(n)^ 2;
 
%% orignal LMS
output_1(n)=WH_filter_1'*input;
e1(n)=clean_audio(n)-output_1(n);
WH_filter_1= 0.8*WH_filter_1+ 0.1*input*e1(n); 
end
final_error=final_error+error;
error=final_error/Loop;
error=error.^ 2;
error(n)=error'*error;
 

output=output(:);
output_1=output_1(:);
e=e(:);
e1=e1(:);
srr=srr(:);
srr1=srr1(:);

 
inp1= reshape(reverb_audio, 691247, 2);

d1= reshape(clean_audio, 691247, 2);
y_nlms= 2* reshape(output, 691247, 2);
y_lms= 2* reshape(output_1, 691247, 2);

error= 10* log10(error);
e_nlms= reshape(e, 691247, 2);
e_lms= reshape(e1, 691247, 2);
error_nlms= reshape(error, 691247, 2);
srr2= reshape(srr, 691247, 2);
srr3= reshape(srr1, 691247, 2);
%% Plot
figure( 1);
subplot( 3, 1, 1);
plot(k,inp1);
xlabel( 'sample');
ylabel( 'audio with reverb');
subplot( 3, 1, 2);
plot(k,y_nlms, 'r');
xlabel( 'sample');
ylabel( 'Improved LMS');
subplot( 3, 1, 3);
plot(k,y_lms, 'g');
axis([ 0 691246 - 1 1]);
xlabel( 'sample');
ylabel( 'LMS');
 
sound(d1,sr);
sound(inp1,sr);
sound(y_nlms,sr);
mp3write(y_nlms,sr, 'nlms.mp3');
sound(y_lms,sr);
mp3write(y_lms,sr, 'lms.mp3');