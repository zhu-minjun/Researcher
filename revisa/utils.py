"""
Utility functions for ai-researcher package.
"""
def get_paper_from_generated_text(generated_text):
    """
    Parse and extract different sections from a generated academic paper text.
    The function expects the text to contain specific sections marked with '##'
    and LaTeX content.

    Args:
        generated_text (str): The full text of the generated paper containing
            marked sections and LaTeX content

    Returns:
        dict: A dictionary containing parsed paper sections and metadata
        None: If parsing fails or the content is not properly formatted
    """

    item = {}
    try:
        # Store the original generated text
        item['generated_text'] = generated_text

        # Extract main sections marked with '##'
        Motivation = generated_text.split('## Motivation')[1].split('## Main Idea')[0]
        Idea = generated_text.split('## Main Idea')[1].split('## Interestingness')[0]
        Interestingness = generated_text.split('## Interestingness')[1].split('## Feasibility')[0]
        Feasibility = generated_text.split('## Feasibility')[1].split('## Novelty')[0]
        Novelty = generated_text.split('## Novelty')[1].split('```latex')[0]

        # Extract and process LaTeX content
        latex = ''
        latex += generated_text.split('```latex')[1].split('```')[0]

        # Parse title and abstract from LaTeX
        title = latex.split(r'\title{')[1].split(r'\begin{abstract}')[0].replace('}','')
        abstract = latex.split(r'\begin{abstract}')[1].split('\end{abstract}')[0]

        # Extract and parse experimental setup
        Experimental_Setup = generated_text.split('## Experimental Setup')[1].split('```json')[1].split('```')[0]
        try:
            # Attempt to parse experimental setup as JSON
            Experimental_Setup = json.loads(Experimental_Setup)
        except:
            pass

        # Extract experimental results, handling two possible formats
        if '## Experimental_results' in generated_text:
            Experimental_results = generated_text.split('## Experimental_results')[1].split('```json')[1].split('```')[
                0]
        else:
            Experimental_results = generated_text.split('## Experimental Setup')[1].split('```json')[2].split('```')[0]

        # Add remaining LaTeX content after experimental results
        latex += generated_text.split(Experimental_results)[1]

        try:
            # Attempt to parse experimental results as JSON
            Experimental_results = json.loads(Experimental_results)
        except:
            pass

        # Process LaTeX content to extract main body
        # Stops at acknowledgment, conclusion, disclosure sections or clearpage
        latex_context = ''
        is_latex = False
        for l in latex.split('\n'):
            if 'section' in l.lower() and 'acknowledgment' in l.lower():
                is_latex = True
                break
            latex_context += l + '\n'
            if 'section' in l.lower() and 'conclusion' in l.lower():
                if is_latex:
                    break
                is_latex = True
            if 'section' in l.lower() and 'disclosure' in l.lower():
                if is_latex:
                    break
                is_latex = True
            if r'\clearpage' in l.lower():
                if is_latex:
                    break
                is_latex = True
                break

        # Populate the return dictionary with all extracted contents
        item['motivation'] = Motivation
        item['idea'] = Idea
        item['interestingness'] = Interestingness
        item['feasibility'] = Feasibility
        item['novelty'] = Novelty
        item['title'] = title
        item['abstract'] = abstract
        item['Experimental_Setup'] = Experimental_Setup
        item['Experimental_results'] = Experimental_results

        # Clean up LaTeX context by removing markdown code markers
        latex_context = latex_context.replace('```latex', '').replace('```json', '').replace('```', '')

        # Return the item only if valid LaTeX content was found
        if is_latex:
            item['latex'] = latex_context
            return item
        else:
            item['latex'] = ''
            return item

    except:
        # Return None if any parsing error occurs
        item['latex'] = ''
        return item


def validate_references(references_file):
    """
    Validate the BibTeX references file.

    Args:
        references_file (str): Path to the BibTeX file

    Returns:
        bool: Whether the references file is valid
    """
    try:
        import bibtexparser
        with open(references_file, 'r', encoding='utf-8') as bibtex_file:
            bib_database = bibtexparser.load(bibtex_file)
        return len(bib_database.entries) > 0
    except Exception as e:
        print(f"References file validation failed: {e}")
        return False


def print_paper_summary(paper):
    """
    Print a summary of the generated research paper.

    Args:
        paper (dict): Generated research paper
    """
    print("📄 Research Paper Summary:")
    print(f"Title: {paper.get('title', 'N/A')}")
    print("\n📝 Abstract:")
    print(paper.get('abstract', 'N/A'))

    print("\n🔍 Key Sections:")
    for key in ['motivation', 'idea',]:
        print(f"\n{key.capitalize()}:")
        print(paper.get(key, 'N/A') + '...' if paper.get(key) else 'N/A')


def get_reviewer_score_7B(generated_text):
    try:
        pred = {}
        reviews = []
        review_rate = []
        rating = []
        summary = []
        soundness = []
        presentation = []
        contribution = []
        strengths = []
        weaknesses = []
        questions = []
        flag_for_ethics_review = []
        confidence = []
        paper_decision = ''
        meta_review = ''
        for review in generated_text.split('**********\n'):
            if review != '':
                if '## Paper Decision\n\n' in review:
                    review, paper_decision = review.split('## Paper Decision\n\n')[:2]
                    paper_decision = paper_decision.split('\n')[0]
                    if 'accept' in paper_decision.lower():
                        paper_decision = 'Accept'
                    else:
                        paper_decision = 'Reject'
                    break
                if '## Meta Review\n\n' in review:
                    review, meta_review = review.split('## Meta Review\n\n')[:2]

                elif '## Summary\n\n' in review and '## Soundness\n\n' in review and '## Presentation\n\n' in review:
                    reviews.append(review)

                    if '#@ Summary\n\n' in review:
                        summary.append(review.split('## Summary\n\n')[1].split('##')[0])
                    else:
                        summary.append('')

                    if '## Soundness\n\n' in review:
                        soundness.append(review.split('## Soundness\n\n')[1].split('##')[0])
                    else:
                        soundness.append('')

                    if '## Presentation\n\n' in review:
                        presentation.append(review.split('## Presentation\n\n')[1].split('##')[0])
                    else:
                        presentation.append('')

                    if '## Contribution\n\n' in review:
                        contribution.append(review.split('## Contribution\n\n')[1].split('##')[0])
                    else:
                        contribution.append('')

                    if '## Strengths\n\n' in review:
                        strengths.append(review.split('## Strengths\n\n')[1].split('##')[0])
                    else:
                        strengths.append('')

                    if '## Weaknesses\n\n' in review:
                        weaknesses.append(review.split('## Weaknesses\n\n')[1].split('##')[0])
                    else:
                        weaknesses.append('')

                    if '## Questions\n\n' in review:
                        questions.append(review.split('## Questions\n\n')[1].split('##')[0])
                    else:
                        questions.append('')

                    if '## Flag For Ethics Review\n\n' in review:
                        flag_for_ethics_review.append(
                            review.split('## Flag For Ethics Review\n\n')[1].split('##')[0])
                    else:
                        flag_for_ethics_review.append('')

                    if '## Rating\n\n' in review:
                        review_rate.append(review.split('## Rating\n\n')[1].split('##')[0])
                        rating.append(float(review.split('## Rating\n\n')[1].split('##')[0][0]))
                    else:
                        review_rate.append('')
                        rating.append(0)

                    if '## Confidence\n\n' in review:
                        confidence.append(review.split('## Confidence\n\n')[1].split('******')[0])
                    else:
                        confidence.append('')
        if paper_decision == '':
            return None
        pred['content'] = generated_text
        pred['reviews'] = reviews
        pred['summary'] = summary
        pred['review_rate'] = review_rate
        pred['rating'] = rating
        pred['soundness'] = soundness
        pred['presentation'] = presentation
        pred['contribution'] = contribution
        pred['strength'] = strengths
        pred['weaknesses'] = weaknesses
        pred['questions'] = questions
        pred['flag_for_ethics_review'] = flag_for_ethics_review
        pred['confidence'] = confidence
        pred['paper_decision'] = paper_decision
        pred['meta_review'] = meta_review
        pred['avg_rating'] = sum(rating) / len(rating)

        return pred
    except:
        return None


def get_reviewer_score_123B(generated_text):
    try:
        pred = {}
        reviews = []
        review_rate = []
        rating = []
        summary = []
        soundness = []
        presentation = []
        contribution = []
        strengths = []
        weaknesses = []
        questions = []
        flag_for_ethics_review = []
        confidence = []
        paper_decision = ''
        meta_review = ''
        for review in generated_text.split('## Reviewer\n'):
            if review != '':
                if '## Paper Decision\n\n' in review:
                    review, paper_decision = review.split('## Paper Decision\n\n')[:2]
                    paper_decision = paper_decision.split('\n')[0]
                    if 'accept' in paper_decision.lower():
                        paper_decision = 'Accept'
                    else:
                        paper_decision = 'Reject'
                if '## Meta Review\n\n' in review:
                    review, meta_review = review.split('## Meta Review\n\n')[:2]
                reviews.append(review)

                if '### Summary\n\n' in review:
                    summary.append(review.split('### Summary\n\n')[1].split('###')[0])
                else:
                    summary.append('')

                if '### Soundness\n\n' in review:
                    soundness.append(review.split('### Soundness\n\n')[1].split('###')[0])
                else:
                    soundness.append('')

                if '### Presentation\n\n' in review:
                    presentation.append(review.split('### Presentation\n\n')[1].split('###')[0])
                else:
                    presentation.append('')

                if '### Contribution\n\n' in review:
                    contribution.append(review.split('### Contribution\n\n')[1].split('###')[0])
                else:
                    contribution.append('')

                if '### Strengths\n\n' in review:
                    strengths.append(review.split('### Strengths\n\n')[1].split('###')[0])
                else:
                    strengths.append('')

                if '### Weaknesses\n\n' in review:
                    weaknesses.append(review.split('### Weaknesses\n\n')[1].split('###')[0])
                else:
                    weaknesses.append('')

                if '### Questions\n\n' in review:
                    questions.append(review.split('### Questions\n\n')[1].split('###')[0])
                else:
                    questions.append('')

                if '### Flag For Ethics Review\n\n' in review:
                    flag_for_ethics_review.append(
                        review.split('### Flag For Ethics Review\n\n')[1].split('###')[0])
                else:
                    flag_for_ethics_review.append('')

                if '### Rating\n\n' in review:
                    review_rate.append(review.split('### Rating\n\n')[1].split('###')[0])
                    rating.append(float(review.split('### Rating\n\n')[1].split('###')[0][0]))
                else:
                    review_rate.append('')
                    rating.append(0)

                if '### Confidence\n\n' in review:
                    confidence.append(review.split('### Confidence\n\n')[1].split('******')[0])
                else:
                    confidence.append('')

        if paper_decision == '':
            return None
        pred['content'] = generated_text
        pred['reviews'] = reviews
        pred['summary'] = summary
        pred['review_rate'] = review_rate
        pred['rating'] = rating
        pred['soundness'] = soundness
        pred['presentation'] = presentation
        pred['contribution'] = contribution
        pred['strength'] = strengths
        pred['weaknesses'] = weaknesses
        pred['questions'] = questions
        pred['flag_for_ethics_review'] = flag_for_ethics_review
        pred['confidence'] = confidence
        pred['paper_decision'] = paper_decision
        pred['meta_review'] = meta_review
        pred['avg_rating'] = sum(rating) / len(rating)

        return pred
    except:
        return None


def get_reviewer_score(generated_text):
    pred = get_reviewer_score_7B(generated_text)
    if pred == None:
        pred = get_reviewer_score_123B(generated_text)
    elif pred['rating'] == 0:
        pred = get_reviewer_score_123B(generated_text)
    return pred



def print_review_summary(review):
    """
    Print a summary of the generated paper review.

    Args:
        review (dict): Generated paper review
    """
    print("📋 Paper Review Summary:")
    print("\n🌟 Overall Assessment:")
    print(review.get('overall_assessment', 'N/A'))

    print("\n💪 Strengths:")
    print(review.get('strengths', 'N/A'))

    print("\n🔧 Weaknesses:")
    print(review.get('weaknesses', 'N/A'))

    print("\n📊 Recommendation:")
    print(review.get('recommendation', 'N/A'))

    print(f"\n💯 Score: {review.get('score', 'N/A')}")
