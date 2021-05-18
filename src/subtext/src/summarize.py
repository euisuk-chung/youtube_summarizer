from src.backbone import WindowEmbedder, Extractor


class SubtextSummarizer:
    '''
    Info:
    Arguments:
        args
        ckpt_path: path to the pre-trained kobertsum weights
    '''
    def __init__(self, args=None, ckpt_path='', input_script=[]):
        self.args = args
        self.ckpt_path = ckpt_path
        self.input_script = ['\n'.join(script) for script in input_script]
    
    
    def summarize_subtexts(self):
        # extractive summary
        extractor = Extractor(args=self.args, use_gpu=True, checkpoint_path=self.ckpt_path)
        
        summary_result = []
        for src in self.input_script:
            summary = extractor.summarize(src, "\n")
            summary_refined = self._sort_summary(summary)
            summary_result.append(summary_refined)
            
        return summary_result
    
    
    def _sort_summary(self, summary_input):
        '''
        Info: Sort summary result in ascending order
        '''
        summary_text = summary_input[0][0].split('. ')
        summary_idx = summary_input[1][0]
        to_sort = list(zip(summary_text, summary_idx))
        
        to_sort.sort(key=lambda x: x[1])
        sorted_summary = [cont[0] for cont in to_sort]
        
        return sorted_summary